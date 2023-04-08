import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

// Adds a list of images that are generated to the bottom of the page
// This script was created by pythongosssss
// https://github.com/pythongosssss/ComfyUI-Custom-Scripts
app.registerExtension({
	name: "Comfy.ImageFeed",
	setup() {
        //CODE HERE
        //create imageList element
        const imageList = document.createElement("div");
        Object.assign(imageList.style, {
            minHeight: "30px",
            maxHeight: "1000px",
            width: "100vw",
            position: "absolute",
            bottom: 0,
            background: "#333",
            overflow: "auto",
            border: "2px solid #333",
            zIndex: "99",
            display: "flex",
            flexWrap: "wrap",
            userSelect: "none",
            alignContent: "baseline"
        });

        // add CSS rules for resize cursor
        const resizeHandle = document.createElement("div");
        Object.assign(resizeHandle.style, {
            position: "absolute",
            top: "-5px",
            right: "0",
            left: "0",
            height: "10px",
            cursor: "row-resize",
            zIndex: "1"
        });
        imageList.appendChild(resizeHandle);

        // add hover style to resize handle
        const hoverStyle = document.createElement("style");
        hoverStyle.innerHTML = `
            .resize-handle:hover {
                background-color: #666;
            }
        `;
        document.head.appendChild(hoverStyle);

        // set class for resize handle
        resizeHandle.classList.add("resize-handle");

        // add mousedown event listener to resize handle
        let startY = 0;
        let startHeight = 0;
        resizeHandle.addEventListener("mousedown", (event) => {
            startY = event.clientY;
            startHeight = parseInt(getComputedStyle(imageList).height);
            document.addEventListener("mousemove", resize);
            document.addEventListener("mouseup", stopResize);
        });

        // resize function
        function resize(event) {
            const newHeight = startHeight + startY - event.clientY;
            imageList.style.height = newHeight + "px";
        }
        var allImages = []

        function loadImages(detail) {
          const images = detail.output.images.filter(
            (img) => img.type === "output" && img.filename !== "_output_images_will_be_put_here"
          );
          allImages.push(...images);
          for (const src of images) {
            const imgContainer = document.createElement("div");
            imgContainer.style.cssText = "height: 120px; width: 120px; position: relative;";

            const imgDelete = document.createElement("button");
            imgDelete.innerHTML = "🗑️";
            imgDelete.style.cssText =
              "position: absolute; top: 0; right: 0; width: 20px; text-indent: -4px; right: 5px; height: 20px; cursor: pointer; position: absolute; top: 5px; font-size: 12px; line-height: 12px;";

            imgDelete.addEventListener("click", async () => {
              const confirmDelete = confirm("Are you sure you want to delete this image?");
              if (confirmDelete) {
                await api.deleteImage(src.filename);
                let newAllImages = allImages.filter(image => image.filename !== src.filename);
                allImages = newAllImages;
                imgContainer.remove();
              }
            });

            const img = document.createElement("img");
            img.setAttribute("filename", src.filename);
            img.style.cssText = "height: 120px; width: 120px; object-fit: cover;";
            img.src = `/view?filename=${encodeURIComponent(src.filename)}&type=${src.type}&subfolder=${encodeURIComponent(src.subfolder)}`;
            img.addEventListener("click", () => {
              const popup = document.createElement("div");
              popup.style.cssText = "position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.5); display: flex; justify-content: center; align-items: center; z-index: 999;";

              const popupImg = document.createElement("img");
              popupImg.src = img.src;
              popupImg.style.cssText = "max-height: 80vh; max-width: 80vw;";
              let currentIndex = allImages.indexOf(src);

              const closeButton = document.createElement("button");
              closeButton.innerHTML = "❌";
              closeButton.style.cssText = "position: absolute; top: 0; right: 0; padding: 5px; font-size: 20px; line-height: 20px; background-color: transparent; border: none; color: white; cursor: pointer;";

              const nextButton = document.createElement("button");
              nextButton.innerHTML = "▶";
              nextButton.style.cssText = "position: absolute; top: 50%; right: 10px; padding: 5px; font-size: 20px; line-height: 20px; background-color: transparent; border: none; color: white; cursor: pointer; transform: translateY(-50%);";

              const prevButton = document.createElement("button");
              prevButton.innerHTML = "◀";
              prevButton.style.cssText = "position: absolute; top: 50%; left: 10px; padding: 5px; font-size: 20px; line-height: 20px; background-color: transparent; border: none; color: white; cursor: pointer; transform: translateY(-50%);";

              closeButton.addEventListener("click", () => {
                popup.remove();
              });
              nextButton.addEventListener("click", () => {
                currentIndex--;
                if (currentIndex < 0) {
                  currentIndex = allImages.length - 1;
                }
                popupImg.src = `/view?filename=${encodeURIComponent(allImages[currentIndex].filename)}&type=${allImages[currentIndex].type}&subfolder=${encodeURIComponent(allImages[currentIndex].subfolder)}`;
              });
              prevButton.addEventListener("click", () => {
                currentIndex++;
                if (currentIndex >= allImages.length) {
                  currentIndex = 0;
                }
                popupImg.src = `/view?filename=${encodeURIComponent(allImages[currentIndex].filename)}&type=${allImages[currentIndex].type}&subfolder=${encodeURIComponent(allImages[currentIndex].subfolder)}`;
              });
              popup.addEventListener("click", (event) => {
                if (event.target === popup) {
                  popup.remove();
                }
              });
              popup.append(popupImg);
              popup.append(closeButton);
              popup.append(nextButton);
              popup.append(prevButton);
              document.body.append(popup);
            });


            imgContainer.append(imgDelete);
            imgContainer.append(img);
            imageList.prepend(imgContainer);
          }
        }

        // stop resize function
        function stopResize() {
            document.removeEventListener("mousemove", resize);
            document.removeEventListener("mouseup", stopResize);
        }

        // append imageList element to document
        document.body.append(imageList);
        const menu = document.createElement("div");
        Object.assign(menu.style, {
            height: "100%",
            width: "90px",
            right:"0px",
            top:"0px"
        });
        imageList.append(menu);
		function makeButton(text, style) {
			const btn = document.createElement("button");
			btn.type = "button";
			btn.textContent = text;
            Object.assign(btn.style, {
              ...style,
              height: "20px",
              width: "80px",
              cursor: "pointer",
              position: "absolute",
              fontSize: "12px",
              lineHeight: "12px",
            });
			menu.append(btn);
			return btn;
		}

		const showButton = document.createElement("button");
		const closeButton = makeButton("❌ Close", {
			textIndent: "-4px",
			top: "5px",
			right: "5px",
		});
		closeButton.onclick = () => {
			imageList.style.display = "none";
			showButton.style.display = "unset";
		};

		const clearButton = makeButton("✖ Clear", {
			top: "30px",
			right: "5px",
		});
		clearButton.onclick = () => {
		    allImages = []
			imageList.replaceChildren(menu, resizeHandle);
		};
		const deleteAllButton = makeButton("🗑️ Delete", {
			top: "55px",
			right: "5px",
		});
		deleteAllButton.onclick = () => {
            const confirmDelete = confirm("Are you sure you want to delete all images?");
            if (confirmDelete) {
                api.deleteAllImages();
                allImages = []
                imageList.replaceChildren(menu, resizeHandle);
			}
		};
        api.getOutput().then(data => {
        try {
                if (data.message == "Success"){
                    var images = data.filenames[0].map((filename) => {
                      return { filename: filename, type: 'output', subfolder: '' };
                    });
                    var output = {images: images}
                    var detail = {output: output}
                    loadImages(detail);
                }
                else  {
                    deleteAllButton.setAttribute("disabled", true);
                }
            } catch(err){
                deleteAllButton.setAttribute("disabled", true);
                console.error(err);
            }
        });
		showButton.classList.add("comfy-settings-btn");
		showButton.style.right = "16px";
		showButton.style.cursor = "pointer";
		showButton.style.display = "none";
		showButton.textContent = "🖼️";
		showButton.onclick = () => {
			imageList.style.display = "flex";
			showButton.style.display = "none";
		};
		document.querySelector(".comfy-settings-btn").after(showButton);

		api.addEventListener("executed", ({ detail }) => {
            loadImages(detail);
		});
	},
});