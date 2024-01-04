/**
 * Changes the background color of the canvas.
 *
 * @method updateBackground
 * @param {image} String
 * @param {clearBackgroundColor} String
 * @
 */
LGraphCanvas.prototype.updateBackground = function (image, clearBackgroundColor) {
	this._bg_img = new Image();
	this._bg_img.name = image;
	this._bg_img.src = image;
	this._bg_img.onload = () => {
		this.draw(true, true);
	};
	this.background_image = image;

	this.clear_background = true;
	this.clear_background_color = clearBackgroundColor;
	this._pattern = null
}
