document.addEventListener('DOMContentLoaded', () => {
    // Attempt to find a known menu element
    let menuElement = document.querySelector(".comfy-menu-buttons");

    if (!menuElement) {
        menuElement = document.querySelector(".comfy-horizontal-menu");
    }

    if (!menuElement) {
        const queueButton = document.getElementById("queue-button");
        if (queueButton && queueButton.parentElement) {
            menuElement = queueButton.parentElement;
        }
    }

    // Fallback: try to find any element with 'menu' in its class or id
    if (!menuElement) {
        const allElements = document.getElementsByTagName('*');
        for (let i = 0; i < allElements.length; i++) {
            const el = allElements[i];
            if ((el.className && typeof el.className === 'string' && el.className.includes('menu')) ||
                (el.id && el.id.includes('menu'))) {
                // Check if it's a plausible candidate (e.g., not too deep, visible)
                // This is a very rough heuristic
                if (el.children.length > 0 && el.children.length < 10 && el.offsetParent !== null) {
                     menuElement = el;
                     console.log("Found a generic menu element:", menuElement);
                     break;
                }
            }
        }
    }


    if (menuElement) {
        const galleryLink = document.createElement('a');
        galleryLink.href = 'gallery.html';
        galleryLink.textContent = 'Gallery';
        galleryLink.id = 'gallery-button'; // Added an ID for easier selection/styling if needed

        // Basic styling to make it look like other buttons if possible
        // This is highly dependent on the existing CSS of the application
        // Attempt to copy styles from an existing button if one exists
        const existingButton = menuElement.querySelector('button') || menuElement.querySelector('a');
        if (existingButton) {
            galleryLink.className = existingButton.className; // Copy class
             // Copy some inline styles if they exist (might not be ideal but can work)
            if (existingButton.style.padding) galleryLink.style.padding = existingButton.style.padding;
            if (existingButton.style.margin) galleryLink.style.margin = existingButton.style.margin;
            if (existingButton.style.textDecoration) galleryLink.style.textDecoration = existingButton.style.textDecoration;
            if (existingButton.style.color) galleryLink.style.color = existingButton.style.color;
            if (existingButton.style.backgroundColor) galleryLink.style.backgroundColor = existingButton.style.backgroundColor;
            if (existingButton.style.border) galleryLink.style.border = existingButton.style.border;
            if (existingButton.style.borderRadius) galleryLink.style.borderRadius = existingButton.style.borderRadius;

        } else {
            // Default minimal styling
            galleryLink.style.padding = '5px 10px';
            galleryLink.style.margin = '0 5px';
            galleryLink.style.textDecoration = 'none';
            galleryLink.style.border = '1px solid #333';
            galleryLink.style.borderRadius = '4px';
            galleryLink.style.color = '#333';
            galleryLink.style.backgroundColor = '#f0f0f0';
        }

        // Specific style for our gallery button if not overridden by copied styles
        if (!galleryLink.style.display) galleryLink.style.display = 'inline-block'; // Ensure it's displayed

        menuElement.appendChild(galleryLink);
        console.log('Gallery link added to menu:', menuElement);
    } else {
        console.warn('Could not find a suitable menu element to add the gallery link.');
        // Fallback: Add it to the body or a prominent header if nothing else is found
        const body = document.body;
        const galleryLink = document.createElement('a');
        galleryLink.href = 'gallery.html';
        galleryLink.textContent = 'Open Gallery';
        galleryLink.style.position = 'fixed';
        galleryLink.style.top = '10px';
        galleryLink.style.right = '10px';
        galleryLink.style.padding = '10px';
        galleryLink.style.backgroundColor = '#007bff';
        galleryLink.style.color = 'white';
        galleryLink.style.textDecoration = 'none';
        galleryLink.style.zIndex = '1000';
        galleryLink.style.border = '1px solid #0056b3'
        galleryLink.style.borderRadius = '5px';
        body.appendChild(galleryLink);
        console.log('Gallery link added as a fallback floating button.');
    }
});
