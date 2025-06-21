document.addEventListener('DOMContentLoaded', () => {
    initGallery();
});

async function initGallery() {
    try {
        const items = await fetchGalleryItems();
        renderGalleryItems(items);
    } catch (error) {
        console.error("Error initializing gallery:", error);
        const galleryContainer = document.getElementById('gallery-container');
        if (galleryContainer) {
            galleryContainer.innerHTML = '<p>Error loading gallery items. Please try again later.</p>';
        }
    }
}

async function fetchGalleryItems() {
    const response = await fetch('/gallery');
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}

function renderGalleryItems(items) {
    const galleryContainer = document.getElementById('gallery-container');
    if (!galleryContainer) {
        console.error("Gallery container not found!");
        return;
    }

    galleryContainer.innerHTML = ''; // Clear previous items

    if (!items || items.length === 0) {
        galleryContainer.innerHTML = '<p>No items in the gallery.</p>';
        return;
    }

    items.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'gallery-item';

        const img = document.createElement('img');
        // Assuming 'item.path' is the correct relative path including any necessary subdirectories
        // and the '.gallery' part of the filename.
        img.src = `userdata/${item.path}`;
        img.alt = item.filename;
        img.className = 'gallery-thumbnail';
        img.onerror = () => { // Basic error handling for broken images
            img.alt = 'Image not found';
            // Optionally, display a placeholder or hide the item
        };

        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove from Gallery';
        removeButton.onclick = async () => {
            try {
                // item.path should be the full relative path including any .gallery part
                const response = await fetch(`/userdata/${item.path}/gallery`, { method: 'POST' });
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(`Failed to remove item: ${errorData.error || response.status}`);
                }
                // Refresh the gallery to show changes
                initGallery();
            } catch (error) {
                console.error('Error removing item from gallery:', error);
                alert(`Error: ${error.message}`);
            }
        };

        itemDiv.appendChild(img);
        itemDiv.appendChild(removeButton);
        galleryContainer.appendChild(itemDiv);
    });
}
