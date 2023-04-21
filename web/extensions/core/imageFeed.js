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
//        const link = document.createElement("link");
//        link.rel = "stylesheet";
//        link.href = "https://cdnjs.cloudflare.com/ajax/libs/sweetalert/2.1.2/sweetalert.min.css";
//        document.head.appendChild(link);
//
//        const script = document.createElement("script");
//        script.src = "https://cdnjs.cloudflare.com/ajax/libs/sweetalert/2.1.2/sweetalert.min.js";
//        document.body.appendChild(script);

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
        var allImages = [];
        var alwaysYes = false;
        var currentIndex = 0;
        function refreshPopup(){
            var currentPopup = document.querySelector('div[type="popup"]');
            if (currentPopup){
                if (currentIndex >= allImages.length){
                    currentIndex--;
                }
                if (allImages.length == 0){
                   currentPopup.remove();
                }
                else{
                    var newFileName = allImages[currentIndex].filename;
                    var newContainer = document.querySelector('div[type="container"]:has(img[filename="'+newFileName+'"])');
                    var newImg = newContainer.querySelector('img');
                    currentPopup.setAttribute("type", "removed");
                    currentPopup.remove();
                    newImg.click();
                }
            }
        }
        function loadImages(detail) {
          const images = detail.output.images.filter(
            (img) => img.type === "output" && img.filename !== "_output_images_will_be_put_here"
          );
          allImages.push(...images);
          for (const src of images) {
            const imgContainer = document.createElement("div");
            imgContainer.setAttribute("type", "container");
            imgContainer.style.cssText = "height: 120px; width: 120px; position: relative;";
            const imgDelete = document.createElement("button");
            imgDelete.innerHTML = "🗑️";
            imgDelete.style.cssText = "position: absolute; top: 0; right: 0; width: 20px; text-indent: -4px; right: 5px; height: 20px; cursor: pointer; position: absolute; top: 5px; font-size: 12px; line-height: 12px;";
            imgDelete.addEventListener("click", async () => {
              if (alwaysYes) {
                deleteImage_func(src);
                refreshPopup();
              } else {
                Swal.fire({
                  title: 'Are you sure?',
                  text: 'Once deleted, you will not be able to recover this image!',
                  icon: 'warning',
                  confirmButtonText: 'Yes',
                  confirmButtonColor: '#3085d6',
                  showCancelButton: true,
                  cancelButtonText: 'No',
                  cancelButtonColor: '#d33',
                  showDenyButton: true,
                  denyButtonText: 'Always Yes',
                  denyButtonColor: '#589bdb',
                  customClass: {
                    popup: 'lgraphcanvas',
                    content: 'lgraphcanvas',
                  }
                }).then((result) => {
                  if (result.isConfirmed) {
                    deleteImage_func(src);
                    refreshPopup();
                  } else if (result.isDenied) {
                    alwaysYes = true;
                    deleteImage_func(src);
                    refreshPopup();
                  }
                })
              }
            });
            const img = document.createElement("img");
            img.setAttribute("filename", src.filename);
            img.style.cssText = "height: 120px; width: 120px; object-fit: cover;";
            img.src = `/view?filename=${encodeURIComponent(src.filename)}&type=${src.type}&subfolder=${encodeURIComponent(src.subfolder)}`;
            img.addEventListener("click", () => {
              const popup = document.createElement("div");
              popup.setAttribute("type", "popup");
              popup.style.cssText = "position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.5); display: flex; justify-content: center; align-items: center; z-index: 999;";

              const popupImg = document.createElement("img");
              popupImg.setAttribute("type", "popupImg");
              popupImg.src = img.src;
              popupImg.style.cssText = "max-height: 80vh; max-width: 80vw;";
              currentIndex = allImages.indexOf(src);

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
                next();
              });
              prevButton.addEventListener("click", () => {
                prev();
              });
              function next(){
                currentIndex--;
                if (currentIndex < 0) {
                  currentIndex = allImages.length - 1;
                }
                popupImg.src = `/view?filename=${encodeURIComponent(allImages[currentIndex].filename)}&type=${allImages[currentIndex].type}&subfolder=${encodeURIComponent(allImages[currentIndex].subfolder)}`;
              }
              function prev(){
                currentIndex++;
                if (currentIndex >= allImages.length) {
                  currentIndex = 0;
                }
                popupImg.src = `/view?filename=${encodeURIComponent(allImages[currentIndex].filename)}&type=${allImages[currentIndex].type}&subfolder=${encodeURIComponent(allImages[currentIndex].subfolder)}`;
              }
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
        document.addEventListener('keydown', function(event) {
            if (event.keyCode === 37) {
                currentIndex++;
                if (currentIndex >= allImages.length) {
                  currentIndex = 0;
                }
                var popupImg = document.querySelector("img[type='popupImg']");
                popupImg.src = `/view?filename=${encodeURIComponent(allImages[currentIndex].filename)}&type=${allImages[currentIndex].type}&subfolder=${encodeURIComponent(allImages[currentIndex].subfolder)}`;
            }
            else if (event.keyCode === 39) {
                currentIndex--;
                if (currentIndex < 0) {
                  currentIndex = allImages.length - 1;
                }
                var popupImg = document.querySelector("img[type='popupImg']");
                popupImg.src = `/view?filename=${encodeURIComponent(allImages[currentIndex].filename)}&type=${allImages[currentIndex].type}&subfolder=${encodeURIComponent(allImages[currentIndex].subfolder)}`;
            } else if (event.keyCode === 46) { // delete
                var currentPopup = document.querySelector('div[type="popup"]');
                if (currentPopup){
                    var srcToSearch = allImages[currentIndex].filename;
                    var imgContainer = document.querySelector('div[type="container"]:has(img[filename="'+srcToSearch+'"])');
                    var imgDelete = imgContainer.querySelector('button');
                    imgDelete.click();
                }
            } else if (event.keyCode === 13) { // enter
                downloadFile(`/view?filename=${encodeURIComponent(allImages[currentIndex].filename)}&type=${allImages[currentIndex].type}&subfolder=${encodeURIComponent(allImages[currentIndex].subfolder)}`, allImages[currentIndex].filename)
            }
        });
        function deleteImage_func(src_val){
            api.deleteImage(src_val.filename);
            let newAllImages = allImages.filter(image => image.filename !== src_val.filename);
            allImages = newAllImages;
            var imgContainer = document.querySelector('div[type="container"]:has(img[filename="'+src_val.filename+'"])');
            imgContainer.remove();
        }
        function downloadFile(url, filename) {
          const link = document.createElement('a');
          link.setAttribute('href', url);
          link.setAttribute('download', filename);
          link.style.display = 'none';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
        function stopResize() {
            document.removeEventListener("mousemove", resize);
            document.removeEventListener("mouseup", stopResize);
        }
        document.body.append(imageList);
        const menu = document.createElement("div");
        Object.assign(menu.style, {
            height: "100%",
            width: "90px",
            right:"0px",
            top:"0px"
        });
        imageList.append(menu);
		function makeButton(text, style, title) {
			const btn = document.createElement("button");
			btn.type = "button";
			btn.textContent = text;
			btn.title = title;
			btn.classList.add("comfy-list");
            Object.assign(btn.style, {
              ...style,
              height: "20px",
              width: "80px",
              cursor: "pointer",
              position: "absolute",
              fontSize: "12px",
              lineHeight: "12px",
              borderRadius: "5px",
              backgroundColor: "#202020"
            });
			menu.append(btn);
			return btn;
		}

		const showButton = document.createElement("button");
		const closeButton = makeButton("❌ Close", {
			textIndent: "-4px",
			top: "5px",
			right: "5px",
		    }, "Hide the image drawer (Open Drawer button will be displayed on main floating menu)"
		);
		closeButton.onclick = () => {
			imageList.style.display = "none";
			showButton.style.display = "unset";
		};

		const clearButton = makeButton("✖ Clear", {
			top: "30px",
			right: "5px",
			}, "Clears all items displayed in image drawer (This won't delete anything, refreshing the page will reload from Output)"
		);
		clearButton.onclick = () => {
		    allImages = []
			imageList.replaceChildren(menu, resizeHandle);
		};
		const deleteAllButton = makeButton("🗑️ Delete", {
			top: "55px",
			right: "5px",
			}, "Delete all items displayed in image drawer (This won't delete the entire output folder)"
		);
		deleteAllButton.onclick = () => {
            Swal.fire({
              title: 'Are you sure?',
              text: 'Delete all the images currently displayed in the drawer',
              icon: 'warning',
              confirmButtonText: 'Yes',
              confirmButtonColor: '#3085d6',
              showCancelButton: true,
              cancelButtonText: 'No',
              cancelButtonColor: '#d33'
            }).then((result) => {
              if (result.isConfirmed) {
                deleteAllImage_func();
              }
            })
        };
        function deleteAllImage_func(){
            api.deleteAllImages(allImages.map(item => item.filename));
            allImages = []
            imageList.replaceChildren(menu, resizeHandle);
        }

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
            } catch(err){
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