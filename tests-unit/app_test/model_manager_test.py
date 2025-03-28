import pytest
import base64
import json
import struct
from io import BytesIO
from PIL import Image
from aiohttp import web
from unittest.mock import patch
from app.model_manager import ModelFileManager
from app.database.models import Base, Model, Tag
from comfy.cli_args import args
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module

@pytest.fixture
def session():
    # Configure in-memory database
    args.database_url = "sqlite:///:memory:"
    
    # Create engine and session factory
    engine = create_engine(args.database_url)
    Session = sessionmaker(bind=engine)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Patch Session factory
    with patch('app.database.db.Session', Session):
        yield Session()
            
    Base.metadata.drop_all(engine)

@pytest.fixture
def model_manager():
    return ModelFileManager()

@pytest.fixture
def app(model_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    model_manager.add_routes(routes)
    app.add_routes(routes)
    return app

async def test_get_model_preview_safetensors(aiohttp_client, app, tmp_path):
    img = Image.new('RGB', (100, 100), 'white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    safetensors_file = tmp_path / "test_model.safetensors"
    header_bytes = json.dumps({
        "__metadata__": {
            "ssmd_cover_images": json.dumps([img_b64])
        }
    }).encode('utf-8')
    length_bytes = struct.pack('<Q', len(header_bytes))
    with open(safetensors_file, 'wb') as f:
        f.write(length_bytes)
        f.write(header_bytes)

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/0/test_model.safetensors')

        # Verify response
        assert response.status == 200
        assert response.content_type == 'image/webp'

        # Verify the response contains valid image data
        img_bytes = BytesIO(await response.read())
        img = Image.open(img_bytes)
        assert img.format
        assert img.format.lower() == 'webp'

        # Clean up
        img.close()

async def test_get_models(aiohttp_client, app, session):
    tag = Tag(name='test_tag')
    model = Model(
        type='checkpoints',
        path='model1.safetensors',
        title='Test Model'
    )
    model.tags.append(tag)
    session.add(tag)
    session.add(model)
    session.commit()

    client = await aiohttp_client(app)
    resp = await client.get('/v2/models')
    assert resp.status == 200
    data = await resp.json()
    assert len(data) == 1
    assert data[0]['path'] == 'model1.safetensors'
    assert len(data[0]['tags']) == 1
    assert data[0]['tags'][0]['name'] == 'test_tag'

async def test_add_model(aiohttp_client, app, session):
    tag = Tag(name='test_tag')
    session.add(tag)
    session.commit()
    tag_id = tag.id

    with patch('app.model_manager.model_processor') as mock_processor:
        with patch('app.model_manager.get_full_path', return_value='/checkpoints/model1.safetensors'):
            client = await aiohttp_client(app)
            resp = await client.post('/v2/models', json={
                'type': 'checkpoints',
                'path': 'model1.safetensors',
                'title': 'Test Model',
                'tags': [tag_id]
            })
            
            assert resp.status == 200
            data = await resp.json()
            assert data['path'] == 'model1.safetensors'
            assert len(data['tags']) == 1
            assert data['tags'][0]['name'] == 'test_tag'

            # Ensure that models are re-processed after adding
            mock_processor.run.assert_called_once()

async def test_delete_model(aiohttp_client, app, session):
    model = Model(
        type='checkpoints',
        path='model1.safetensors',
        title='Test Model'
    )
    session.add(model)
    session.commit()
    
    with patch('app.model_manager.get_full_path', return_value=None):
        client = await aiohttp_client(app)
        resp = await client.delete('/v2/models?type=checkpoints&path=model1.safetensors')
        assert resp.status == 204
        
        # Verify model was deleted
        model = session.query(Model).first()
        assert model is None

async def test_delete_model_file_exists(aiohttp_client, app, session):
    model = Model(
        type='checkpoints',
        path='model1.safetensors',
        title='Test Model'
    )
    session.add(model)
    session.commit()
    
    with patch('app.model_manager.get_full_path', return_value='/checkpoints/model1.safetensors'):
        client = await aiohttp_client(app)
        resp = await client.delete('/v2/models?type=checkpoints&path=model1.safetensors')
        assert resp.status == 400
        
        data = await resp.json()
        assert "file exists" in data["error"].lower()
        
        # Verify model was not deleted
        model = session.query(Model).first()
        assert model is not None
        assert model.path == 'model1.safetensors'

async def test_get_tags(aiohttp_client, app, session):
    tags = [Tag(name='tag1'), Tag(name='tag2')]
    for tag in tags:
        session.add(tag)
    session.commit()

    client = await aiohttp_client(app)
    resp = await client.get('/v2/tags')
    assert resp.status == 200
    data = await resp.json()
    assert len(data) == 2
    assert {t['name'] for t in data} == {'tag1', 'tag2'}

async def test_create_tag(aiohttp_client, app, session):
    client = await aiohttp_client(app)
    resp = await client.post('/v2/tags', json={'name': 'new_tag'})
    assert resp.status == 200
    data = await resp.json()
    assert data['name'] == 'new_tag'
    
    # Verify tag was created
    tag = session.query(Tag).first()
    assert tag.name == 'new_tag'

async def test_delete_tag(aiohttp_client, app, session):
    tag = Tag(name='test_tag')
    session.add(tag)
    session.commit()
    tag_id = tag.id

    client = await aiohttp_client(app)
    resp = await client.delete(f'/v2/tags?id={tag_id}')
    assert resp.status == 204
    
    # Verify tag was deleted
    tag = session.query(Tag).first()
    assert tag is None

async def test_add_model_tag(aiohttp_client, app, session):
    tag = Tag(name='test_tag')
    model = Model(
        type='checkpoints',
        path='model1.safetensors',
        title='Test Model'
    )
    session.add(tag)
    session.add(model)
    session.commit()
    tag_id = tag.id

    client = await aiohttp_client(app)
    resp = await client.post('/v2/models/tags', json={
        'tag': tag_id,
        'type': 'checkpoints',
        'path': 'model1.safetensors'
    })
    assert resp.status == 200
    data = await resp.json()
    assert len(data['tags']) == 1
    assert data['tags'][0]['name'] == 'test_tag'

async def test_delete_model_tag(aiohttp_client, app, session):
    tag = Tag(name='test_tag')
    model = Model(
        type='checkpoints',
        path='model1.safetensors',
        title='Test Model'
    )
    model.tags.append(tag)
    session.add(tag)
    session.add(model)
    session.commit()
    tag_id = tag.id

    client = await aiohttp_client(app)
    resp = await client.delete(f'/v2/models/tags?tag={tag_id}&type=checkpoints&path=model1.safetensors')
    assert resp.status == 204
    
    # Verify tag was removed
    model = session.query(Model).first()
    assert len(model.tags) == 0

async def test_add_model_duplicate(aiohttp_client, app, session):
    model = Model(
        type='checkpoints',
        path='model1.safetensors',
        title='Test Model'
    )
    session.add(model)
    session.commit()
    
    with patch('app.model_manager.get_full_path', return_value='/checkpoints/model1.safetensors'):
        client = await aiohttp_client(app)
        resp = await client.post('/v2/models', json={
            'type': 'checkpoints',
            'path': 'model1.safetensors',
            'title': 'Duplicate Model'
        })
        assert resp.status == 400

async def test_add_model_missing_fields(aiohttp_client, app, session):
    client = await aiohttp_client(app)
    resp = await client.post('/v2/models', json={})
    assert resp.status == 400

async def test_add_tag_missing_name(aiohttp_client, app, session):
    client = await aiohttp_client(app)
    resp = await client.post('/v2/tags', json={})
    assert resp.status == 400

async def test_delete_model_not_found(aiohttp_client, app, session):
    client = await aiohttp_client(app)
    resp = await client.delete('/v2/models?type=checkpoints&path=nonexistent.safetensors')
    assert resp.status == 404

async def test_delete_tag_not_found(aiohttp_client, app, session):
    client = await aiohttp_client(app)
    resp = await client.delete('/v2/tags?id=999')
    assert resp.status == 404

async def test_add_model_missing_path(aiohttp_client, app, session):
    client = await aiohttp_client(app)
    resp = await client.post('/v2/models', json={
        'type': 'checkpoints',
        'title': 'Test Model'
    })
    assert resp.status == 400
    data = await resp.json()
    assert "path" in data["error"].lower()

async def test_add_model_invalid_field(aiohttp_client, app, session):
    client = await aiohttp_client(app)
    resp = await client.post('/v2/models', json={
        'type': 'checkpoints',
        'path': 'model1.safetensors',
        'invalid_field': 'some value'
    })
    assert resp.status == 400
    data = await resp.json()
    assert "invalid field" in data["error"].lower()

async def test_add_model_nonexistent_file(aiohttp_client, app, session):
    with patch('app.model_manager.get_full_path', return_value=None):
        client = await aiohttp_client(app)
        resp = await client.post('/v2/models', json={
            'type': 'checkpoints',
            'path': 'nonexistent.safetensors'
        })
        assert resp.status == 404
        data = await resp.json()
        assert "file" in data["error"].lower()

async def test_add_model_invalid_tag(aiohttp_client, app, session):
    with patch('app.model_manager.get_full_path', return_value='/checkpoints/model1.safetensors'):
        client = await aiohttp_client(app)
        resp = await client.post('/v2/models', json={
            'type': 'checkpoints',
            'path': 'model1.safetensors',
            'tags': [999]  # Non-existent tag ID
        })
        assert resp.status == 404
        data = await resp.json()
        assert "tag" in data["error"].lower()

async def test_add_tag_to_nonexistent_model(aiohttp_client, app, session):
    # Create a tag but no model
    tag = Tag(name='test_tag')
    session.add(tag)
    session.commit()
    tag_id = tag.id

    client = await aiohttp_client(app)
    resp = await client.post('/v2/models/tags', json={
        'tag': tag_id,
        'type': 'checkpoints',
        'path': 'nonexistent.safetensors'
    })
    assert resp.status == 404
    data = await resp.json()
    assert "model" in data["error"].lower()

async def test_delete_model_tag_invalid_tag_id(aiohttp_client, app, session):
    # Create a model first
    model = Model(
        type='checkpoints',
        path='model1.safetensors',
        title='Test Model'
    )
    session.add(model)
    session.commit()

    client = await aiohttp_client(app)
    resp = await client.delete('/v2/models/tags?tag=not_a_number&type=checkpoint&path=model1.safetensors')
    assert resp.status == 400
    data = await resp.json()
    assert "invalid tag id" in data["error"].lower()

