from __future__ import annotations
import datetime
import dataclasses
import io
import types
import typing
import uuid

import functools
import typing_extensions

from comfy.api import exceptions
from comfy.api.configurations import schema_configuration

from . import validation

_T_co = typing.TypeVar("_T_co", covariant=True)


class SequenceNotStr(typing.Protocol[_T_co]):
    """
    if a Protocol would define the interface of Sequence, this protocol
    would NOT allow str/bytes as their __contains__ methods are incompatible with the definition in Sequence
    methods from: https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection
    """
    def __contains__(self, value: object, /) -> bool:
        raise NotImplementedError

    def __getitem__(self, index, /):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> typing.Iterator[_T_co]:
        raise NotImplementedError

    def __reversed__(self, /) -> typing.Iterator[_T_co]:
        raise NotImplementedError

none_type_ = type(None)
T = typing.TypeVar('T', bound=typing.Mapping)
U = typing.TypeVar('U', bound=SequenceNotStr)
W = typing.TypeVar('W')


class SchemaTyped:
    additional_properties: typing.Type[Schema]
    all_of: typing.Tuple[typing.Type[Schema], ...]
    any_of: typing.Tuple[typing.Type[Schema], ...]
    discriminator: typing.Mapping[str, typing.Mapping[str, typing.Type[Schema]]]
    default: typing.Union[str, int, float, bool, None]
    enum_value_to_name: typing.Mapping[typing.Union[int, float, str, Bool, None], str]
    exclusive_maximum: typing.Union[int, float]
    exclusive_minimum: typing.Union[int, float]
    format: str
    inclusive_maximum: typing.Union[int, float]
    inclusive_minimum: typing.Union[int, float]
    items: typing.Type[Schema]
    max_items: int
    max_length: int
    max_properties: int
    min_items: int
    min_length: int
    min_properties: int
    multiple_of: typing.Union[int, float]
    not_: typing.Type[Schema]
    one_of: typing.Tuple[typing.Type[Schema], ...]
    pattern: validation.PatternInfo
    properties: typing.Mapping[str, typing.Type[Schema]]
    required: typing.FrozenSet[str]
    types: typing.FrozenSet[typing.Type]
    unique_items: bool


class FileIO(io.FileIO):
    """
    A class for storing files
    Note: this class is not immutable
    """

    def __new__(cls, arg: typing.Union[io.FileIO, io.BufferedReader]):
        if isinstance(arg, (io.FileIO, io.BufferedReader)):
            if arg.closed:
                raise exceptions.ApiValueError('Invalid file state; file is closed and must be open')
            arg.close()
            inst = super(FileIO, cls).__new__(cls, arg.name) # type: ignore
            super(FileIO, inst).__init__(arg.name)
            return inst
        raise exceptions.ApiValueError('FileIO must be passed arg which contains the open file')

    def __init__(self, arg: typing.Union[io.FileIO, io.BufferedReader]):
        """
        Needed for instantiation when passing in arguments of the above type
        """
        pass


class classproperty(typing.Generic[W]):
    def __init__(self, method: typing.Callable[..., W]):
        self.__method = method
        functools.update_wrapper(self, method) # type: ignore

    def __get__(self, obj, cls=None) -> W:
        if cls is None:
            cls = type(obj)
        return self.__method(cls)


class Bool:
    _instances: typing.Dict[typing.Tuple[type, bool], Bool] = {}
    """
    This class is needed to replace bool during validation processing
    json schema requires that 0 != False and 1 != True
    python implementation defines 0 == False and 1 == True
    To meet the json schema requirements, all bool instances are replaced with Bool singletons
    during validation only, and then bool values are returned from validation
    """

    def __new__(cls, arg_: bool, **kwargs):
        """
        Method that implements singleton
        cls base classes: BoolClass, NoneClass, str, decimal.Decimal
        The 3rd key is used in the tuple below for a corner case where an enum contains integer 1
        However 1.0  can also be ingested into that enum schema because 1.0 == 1 and
        Decimal('1.0') == Decimal('1')
        But if we omitted the 3rd value in the key, then Decimal('1.0') would be stored as Decimal('1')
        and json serializing that instance would be '1' rather than the expected '1.0'
        Adding the 3rd value, the str of arg_ ensures that 1.0 -> Decimal('1.0') which is serialized as 1.0
        """
        key = (cls, arg_)
        if key not in cls._instances:
            inst = super().__new__(cls)
            cls._instances[key] = inst
        return cls._instances[key]

    def __repr__(self):
        if bool(self):
            return f'<Bool: True>'
        return f'<Bool: False>'

    @classproperty
    def TRUE(cls):
        return cls(True) # type: ignore

    @classproperty
    def FALSE(cls):
        return cls(False) # type: ignore

    @functools.lru_cache()
    def __bool__(self) -> bool:
        for key, instance in self._instances.items():
            if self is instance:
                return bool(key[1])
        raise ValueError('Unable to find the boolean value of this instance')


def cast_to_allowed_types(
    arg: typing.Union[
        dict,
        validation.immutabledict,
        list,
        tuple,
        float,
        int,
        str,
        datetime.date,
        datetime.datetime,
        uuid.UUID,
        bool,
        None,
        bytes,
        io.FileIO,
        io.BufferedReader,
    ],
    from_server: bool,
    validated_path_to_schemas: typing.Dict[typing.Tuple[typing.Union[str, int], ...], typing.Set[typing.Union[str, int, float, bool, None, validation.immutabledict, tuple]]],
    path_to_item: typing.Tuple[typing.Union[str, int], ...],
    path_to_type: typing.Dict[typing.Tuple[typing.Union[str, int], ...], type]
) -> typing.Union[
    validation.immutabledict,
    tuple,
    float,
    int,
    str,
    bytes,
    Bool,
    None,
    FileIO
]:
    """
    Casts the input payload arg into the allowed types
    The input validated_path_to_schemas is mutated by running this function

    When from_server is False then
    - date/datetime is cast to str
    - int/float is cast to Decimal

    If a Schema instance is passed in it is converted back to a primitive instance because
    One may need to validate that data to the original Schema class AND additional different classes
    those additional classes will need to be added to the new manufactured class for that payload
    If the code didn't do this and kept the payload as a Schema instance it would fail to validate to other
    Schema classes and the code wouldn't be able to mfg a new class that includes all valid schemas
    TODO: store the validated schema classes in validation_metadata

    Args:
        arg: the payload
        from_server: whether this payload came from the server or not
        validated_path_to_schemas: a dict that stores the validated classes at any path location in the payload
    """
    type_error = exceptions.ApiTypeError(f"Invalid type. Required value type is str and passed type was {type(arg)} at {path_to_item}")
    if isinstance(arg, str):
        path_to_type[path_to_item] = str
        return str(arg)
    elif isinstance(arg, (dict, validation.immutabledict)):
        path_to_type[path_to_item] = validation.immutabledict
        return validation.immutabledict(
            {
                key: cast_to_allowed_types(
                    val,
                    from_server,
                    validated_path_to_schemas,
                    path_to_item + (key,),
                    path_to_type,
                )
                for key, val in arg.items()
            }
        )
    elif isinstance(arg, bool):
        """
        this check must come before isinstance(arg, (int, float))
        because isinstance(True, int) is True
        """
        path_to_type[path_to_item] = Bool
        if arg:
            return Bool.TRUE
        return Bool.FALSE
    elif isinstance(arg, int):
        path_to_type[path_to_item] = int
        return arg
    elif isinstance(arg, float):
        path_to_type[path_to_item] = float
        return arg
    elif isinstance(arg, (tuple, list)):
        path_to_type[path_to_item] = tuple
        return tuple(
            [
                cast_to_allowed_types(
                    item,
                    from_server,
                    validated_path_to_schemas,
                    path_to_item + (i,),
                    path_to_type,
                )
                for i, item in enumerate(arg)
            ]
        )
    elif arg is None:
        path_to_type[path_to_item] = type(None)
        return None
    elif isinstance(arg, (datetime.date, datetime.datetime)):
        path_to_type[path_to_item] = str
        if not from_server:
            return arg.isoformat()
        raise type_error
    elif isinstance(arg, uuid.UUID):
        path_to_type[path_to_item] = str
        if not from_server:
            return str(arg)
        raise type_error
    elif isinstance(arg, bytes):
        path_to_type[path_to_item] = bytes
        return bytes(arg)
    elif isinstance(arg, (io.FileIO, io.BufferedReader)):
        path_to_type[path_to_item] = FileIO
        return FileIO(arg)
    raise exceptions.ApiTypeError('Invalid type passed in got input={} type={}'.format(arg, type(arg)))


class SingletonMeta(type):
    """
    A singleton class for schemas
    Schemas are frozen classes that are never instantiated with init args
    All args come from defaults
    """
    _instances: typing.Dict[type, typing.Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Schema(typing.Generic[T, U], validation.SchemaValidator, metaclass=SingletonMeta):

    @classmethod
    def __get_path_to_schemas(
        cls,
        arg,
        validation_metadata: validation.ValidationMetadata,
        path_to_type: typing.Dict[typing.Tuple[typing.Union[str, int], ...], typing.Type]
    ) -> typing.Dict[typing.Tuple[typing.Union[str, int], ...], typing.Type[Schema]]:
        """
        Run all validations in the json schema and return a dict of
        json schema to tuple of validated schemas
        """
        _path_to_schemas: validation.PathToSchemasType = {}
        if validation_metadata.validation_ran_earlier(cls):
            validation.add_deeper_validated_schemas(validation_metadata, _path_to_schemas)
        else:
            other_path_to_schemas = cls._validate(arg, validation_metadata=validation_metadata)
            validation.update(_path_to_schemas, other_path_to_schemas)
        # loop through it make a new class for each entry
        # do not modify the returned result because it is cached and we would be modifying the cached value
        path_to_schemas: typing.Dict[typing.Tuple[typing.Union[str, int], ...], typing.Type[Schema]] = {}
        for path, schema_classes in _path_to_schemas.items():
            schema = typing.cast(typing.Type[Schema], tuple(schema_classes)[-1])
            path_to_schemas[path] = schema
        """
        For locations that validation did not check
        the code still needs to store type + schema information for instantiation
        All of those schemas will be UnsetAnyTypeSchema
        """
        missing_paths = path_to_type.keys() - path_to_schemas.keys()
        for missing_path in missing_paths:
            path_to_schemas[missing_path] = UnsetAnyTypeSchema

        return path_to_schemas

    @staticmethod
    def __get_items(
        arg: tuple,
        path_to_item: typing.Tuple[typing.Union[str, int], ...],
        path_to_schemas: typing.Dict[typing.Tuple[typing.Union[str, int], ...], typing.Type[Schema]]
    ):
        '''
        Schema __get_items
        '''
        cast_items = []

        for i, value in enumerate(arg):
            item_path_to_item = path_to_item + (i,)
            item_cls = path_to_schemas[item_path_to_item]
            new_value = item_cls._get_new_instance_without_conversion(
                value,
                item_path_to_item,
                path_to_schemas
            )
            cast_items.append(new_value)

        return tuple(cast_items)

    @staticmethod
    def __get_properties(
        arg: validation.immutabledict[str, typing.Any],
        path_to_item: typing.Tuple[typing.Union[str, int], ...],
        path_to_schemas: typing.Dict[typing.Tuple[typing.Union[str, int], ...], typing.Type[Schema]]
    ):
        """
        Schema __get_properties, this is how properties are set
        These values already passed validation
        """
        dict_items = {}

        for property_name_js, value in arg.items():
            property_path_to_item = path_to_item + (property_name_js,)
            property_cls = path_to_schemas[property_path_to_item]
            new_value = property_cls._get_new_instance_without_conversion(
                value,
                property_path_to_item,
                path_to_schemas
            )
            dict_items[property_name_js] = new_value

        return validation.immutabledict(dict_items)

    @classmethod
    def _get_new_instance_without_conversion(
        cls,
        arg: typing.Union[int, float, None, Bool, str, validation.immutabledict, tuple, FileIO, bytes],
        path_to_item: typing.Tuple[typing.Union[str, int], ...],
        path_to_schemas: typing.Dict[typing.Tuple[typing.Union[str, int], ...], typing.Type[Schema]]
    ):
        # We have a Dynamic class and we are making an instance of it
        if isinstance(arg, validation.immutabledict):
            used_arg = cls.__get_properties(arg, path_to_item, path_to_schemas)
        elif isinstance(arg, tuple):
            used_arg = cls.__get_items(arg, path_to_item, path_to_schemas)
        elif isinstance(arg, Bool):
            return bool(arg)
        else:
            """
            str, int, float, FileIO, bytes
            FileIO = openapi binary type and the user inputs a file
            bytes = openapi binary type and the user inputs bytes
            """
            return arg
        arg_type = type(arg)
        type_to_output_cls = cls.__get_type_to_output_cls()
        if type_to_output_cls is None:
            return used_arg
        if arg_type not in type_to_output_cls:
            return used_arg
        output_cls = type_to_output_cls[arg_type]
        if arg_type is tuple:
            inst = super(output_cls, output_cls).__new__(output_cls, used_arg) # type: ignore
            inst = typing.cast(U, inst)
            return inst
        assert issubclass(output_cls, validation.immutabledict)
        inst = super(output_cls, output_cls).__new__(output_cls, used_arg) # type: ignore
        inst = typing.cast(T, inst)
        return inst

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: None,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> None: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: typing.Literal[True],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> typing.Literal[True]: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: typing.Literal[False],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> typing.Literal[False]: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: bool,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> bool: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: int,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> int: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: float,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> float: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: typing.Union[str, datetime.date, datetime.datetime, uuid.UUID],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> str: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: SequenceNotStr[INPUT_TYPES_ALL],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> U: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: U,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> U: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: typing.Mapping[str, object],  # object needed as value type for typeddict inputs
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> T: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: typing.Union[
            typing.Mapping[str, INPUT_TYPES_ALL],
            T
        ],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> T: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: typing.Union[io.FileIO, io.BufferedReader],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> FileIO: ...

    @typing.overload
    @classmethod
    def validate_base(
        cls,
        arg: bytes,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> bytes: ...

    @classmethod
    def validate_base(
        cls,
        arg,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None,
    ):
        """
        Schema validate_base

        Args:
            arg (int/float/str/list/tuple/dict/validation.immutabledict/bool/None): the value
            configuration: contains the schema_configuration.SchemaConfiguration that enables json schema validation keywords
                like minItems, minLength etc
        """
        if isinstance(arg, (tuple, validation.immutabledict)):
            type_to_output_cls = cls.__get_type_to_output_cls()
            if type_to_output_cls is not None:
                for output_cls in type_to_output_cls.values():
                    if isinstance(arg, output_cls):
                        # U + T use case, don't run validations twice
                        return arg

        from_server = False
        validated_path_to_schemas: typing.Dict[
            typing.Tuple[typing.Union[str, int], ...],
            typing.Set[typing.Union[str, int, float, bool, None, validation.immutabledict, tuple]]
        ] = {}
        path_to_type: typing.Dict[typing.Tuple[typing.Union[str, int], ...], type] = {}
        cast_arg = cast_to_allowed_types(
            arg, from_server, validated_path_to_schemas, ('args[0]',), path_to_type)
        validation_metadata = validation.ValidationMetadata(
            path_to_item=('args[0]',),
            configuration=configuration or schema_configuration.SchemaConfiguration(),
            validated_path_to_schemas=validation.immutabledict(validated_path_to_schemas)
        )
        path_to_schemas = cls.__get_path_to_schemas(cast_arg, validation_metadata, path_to_type)
        return cls._get_new_instance_without_conversion(
            cast_arg,
            validation_metadata.path_to_item,
            path_to_schemas,
        )

    @classmethod
    def __get_type_to_output_cls(cls) -> typing.Optional[typing.Mapping[type, type]]:
        type_to_output_cls = getattr(cls(), 'type_to_output_cls', None)
        type_to_output_cls = typing.cast(typing.Optional[typing.Mapping[type, type]], type_to_output_cls)
        return type_to_output_cls


def get_class(
    item_cls: typing.Union[types.FunctionType, staticmethod, typing.Type[Schema]],
    local_namespace: typing.Optional[dict] = None
) -> typing.Type[Schema]:
    if isinstance(item_cls, typing._GenericAlias): # type: ignore
        # petstore_api.schemas.StrSchema[~U] -> petstore_api.schemas.StrSchema
        origin_cls = typing.get_origin(item_cls)
        if origin_cls is None:
            raise ValueError('origin class must not be None')
        return origin_cls
    elif isinstance(item_cls, types.FunctionType):
        # referenced schema
        return item_cls()
    elif isinstance(item_cls, staticmethod):
        # referenced schema
        return item_cls.__func__()
    elif isinstance(item_cls, type):
        return item_cls
    elif isinstance(item_cls, typing.ForwardRef):
        return item_cls._evaluate(None, local_namespace)
    raise ValueError('invalid class value passed in')


@dataclasses.dataclass(frozen=True)
class AnyTypeSchema(Schema[T, U]):
    # Python representation of a schema defined as true or {}

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: None,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> None: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: typing.Literal[True],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> typing.Literal[True]: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: typing.Literal[False],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> typing.Literal[False]: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: bool,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> bool: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: int,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> int: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: float,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> float: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: typing.Union[str, datetime.date, datetime.datetime, uuid.UUID],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> str: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: SequenceNotStr[INPUT_TYPES_ALL],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> U: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: U,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> U: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: typing.Union[
            typing.Mapping[str, INPUT_TYPES_ALL],
            T
        ],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> T: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: typing.Union[io.FileIO, io.BufferedReader],
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> FileIO: ...

    @typing.overload
    @classmethod
    def validate(
        cls,
        arg: bytes,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None
    ) -> bytes: ...

    @classmethod
    def validate(
        cls,
        arg,
        configuration: typing.Optional[schema_configuration.SchemaConfiguration] = None,
    ):
        return cls.validate_base(
            arg,
            configuration=configuration
        )

class UnsetAnyTypeSchema(AnyTypeSchema[T, U]):
    # Used when additionalProperties/items was not explicitly defined and a defining schema is needed
    pass

INPUT_TYPES_ALL = typing.Union[
    dict,
    validation.immutabledict,
    typing.Mapping[str, object],  # for TypedDict
    list,
    tuple,
    float,
    int,
    str,
    datetime.date,
    datetime.datetime,
    uuid.UUID,
    bool,
    None,
    bytes,
    io.FileIO,
    io.BufferedReader,
    FileIO
]