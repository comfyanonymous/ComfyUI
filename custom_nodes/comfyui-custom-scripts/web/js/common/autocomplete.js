import { $el } from "../../../../scripts/ui.js";
import { addStylesheet } from "./utils.js";

addStylesheet(import.meta.url);

/*
    https://github.com/component/textarea-caret-position
    The MIT License (MIT)

    Copyright (c) 2015 Jonathan Ong me@jongleberry.com

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
const getCaretCoordinates = (function () {
	// We'll copy the properties below into the mirror div.
	// Note that some browsers, such as Firefox, do not concatenate properties
	// into their shorthand (e.g. padding-top, padding-bottom etc. -> padding),
	// so we have to list every single property explicitly.
	var properties = [
		"direction", // RTL support
		"boxSizing",
		"width", // on Chrome and IE, exclude the scrollbar, so the mirror div wraps exactly as the textarea does
		"height",
		"overflowX",
		"overflowY", // copy the scrollbar for IE

		"borderTopWidth",
		"borderRightWidth",
		"borderBottomWidth",
		"borderLeftWidth",
		"borderStyle",

		"paddingTop",
		"paddingRight",
		"paddingBottom",
		"paddingLeft",

		// https://developer.mozilla.org/en-US/docs/Web/CSS/font
		"fontStyle",
		"fontVariant",
		"fontWeight",
		"fontStretch",
		"fontSize",
		"fontSizeAdjust",
		"lineHeight",
		"fontFamily",

		"textAlign",
		"textTransform",
		"textIndent",
		"textDecoration", // might not make a difference, but better be safe

		"letterSpacing",
		"wordSpacing",

		"tabSize",
		"MozTabSize",
	];

	var isBrowser = typeof window !== "undefined";
	var isFirefox = isBrowser && window.mozInnerScreenX != null;

	return function getCaretCoordinates(element, position, options) {
		if (!isBrowser) {
			throw new Error("textarea-caret-position#getCaretCoordinates should only be called in a browser");
		}

		var debug = (options && options.debug) || false;
		if (debug) {
			var el = document.querySelector("#input-textarea-caret-position-mirror-div");
			if (el) el.parentNode.removeChild(el);
		}

		// The mirror div will replicate the textarea's style
		var div = document.createElement("div");
		div.id = "input-textarea-caret-position-mirror-div";
		document.body.appendChild(div);

		var style = div.style;
		var computed = window.getComputedStyle ? window.getComputedStyle(element) : element.currentStyle; // currentStyle for IE < 9
		var isInput = element.nodeName === "INPUT";

		// Default textarea styles
		style.whiteSpace = "pre-wrap";
		if (!isInput) style.wordWrap = "break-word"; // only for textarea-s

		// Position off-screen
		style.position = "absolute"; // required to return coordinates properly
		if (!debug) style.visibility = "hidden"; // not 'display: none' because we want rendering

		// Transfer the element's properties to the div
		properties.forEach(function (prop) {
			if (isInput && prop === "lineHeight") {
				// Special case for <input>s because text is rendered centered and line height may be != height
				if (computed.boxSizing === "border-box") {
					var height = parseInt(computed.height);
					var outerHeight =
						parseInt(computed.paddingTop) +
						parseInt(computed.paddingBottom) +
						parseInt(computed.borderTopWidth) +
						parseInt(computed.borderBottomWidth);
					var targetHeight = outerHeight + parseInt(computed.lineHeight);
					if (height > targetHeight) {
						style.lineHeight = height - outerHeight + "px";
					} else if (height === targetHeight) {
						style.lineHeight = computed.lineHeight;
					} else {
						style.lineHeight = 0;
					}
				} else {
					style.lineHeight = computed.height;
				}
			} else {
				style[prop] = computed[prop];
			}
		});

		if (isFirefox) {
			// Firefox lies about the overflow property for textareas: https://bugzilla.mozilla.org/show_bug.cgi?id=984275
			if (element.scrollHeight > parseInt(computed.height)) style.overflowY = "scroll";
		} else {
			style.overflow = "hidden"; // for Chrome to not render a scrollbar; IE keeps overflowY = 'scroll'
		}

		div.textContent = element.value.substring(0, position);
		// The second special handling for input type="text" vs textarea:
		// spaces need to be replaced with non-breaking spaces - http://stackoverflow.com/a/13402035/1269037
		if (isInput) div.textContent = div.textContent.replace(/\s/g, "\u00a0");

		var span = document.createElement("span");
		// Wrapping must be replicated *exactly*, including when a long word gets
		// onto the next line, with whitespace at the end of the line before (#7).
		// The  *only* reliable way to do that is to copy the *entire* rest of the
		// textarea's content into the <span> created at the caret position.
		// For inputs, just '.' would be enough, but no need to bother.
		span.textContent = element.value.substring(position) || "."; // || because a completely empty faux span doesn't render at all
		div.appendChild(span);

		var coordinates = {
			top: span.offsetTop + parseInt(computed["borderTopWidth"]),
			left: span.offsetLeft + parseInt(computed["borderLeftWidth"]),
			height: parseInt(computed["lineHeight"]),
		};

		if (debug) {
			span.style.backgroundColor = "#aaa";
		} else {
			document.body.removeChild(div);
		}

		return coordinates;
	};
})();

/*
    Key functions from:
    https://github.com/yuku/textcomplete
    © Yuku Takahashi - This software is licensed under the MIT license.

    The MIT License (MIT)

    Copyright (c) 2015 Jonathan Ong me@jongleberry.com

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
const CHAR_CODE_ZERO = "0".charCodeAt(0);
const CHAR_CODE_NINE = "9".charCodeAt(0);

class TextAreaCaretHelper {
	constructor(el, getScale) {
		this.el = el;
		this.getScale = getScale;
	}

	#calculateElementOffset() {
		const rect = this.el.getBoundingClientRect();
		const owner = this.el.ownerDocument;
		if (owner == null) {
			throw new Error("Given element does not belong to document");
		}
		const { defaultView, documentElement } = owner;
		if (defaultView == null) {
			throw new Error("Given element does not belong to window");
		}
		const offset = {
			top: rect.top + defaultView.pageYOffset,
			left: rect.left + defaultView.pageXOffset,
		};
		if (documentElement) {
			offset.top -= documentElement.clientTop;
			offset.left -= documentElement.clientLeft;
		}
		return offset;
	}

	#isDigit(charCode) {
		return CHAR_CODE_ZERO <= charCode && charCode <= CHAR_CODE_NINE;
	}

	#getLineHeightPx() {
		const computedStyle = getComputedStyle(this.el);
		const lineHeight = computedStyle.lineHeight;
		// If the char code starts with a digit, it is either a value in pixels,
		// or unitless, as per:
		// https://drafts.csswg.org/css2/visudet.html#propdef-line-height
		// https://drafts.csswg.org/css2/cascade.html#computed-value
		if (this.#isDigit(lineHeight.charCodeAt(0))) {
			const floatLineHeight = parseFloat(lineHeight);
			// In real browsers the value is *always* in pixels, even for unit-less
			// line-heights. However, we still check as per the spec.
			return this.#isDigit(lineHeight.charCodeAt(lineHeight.length - 1))
				? floatLineHeight * parseFloat(computedStyle.fontSize)
				: floatLineHeight;
		}
		// Otherwise, the value is "normal".
		// If the line-height is "normal", calculate by font-size
		return this.#calculateLineHeightPx(this.el.nodeName, computedStyle);
	}

	/**
	 * Returns calculated line-height of the given node in pixels.
	 */
	#calculateLineHeightPx(nodeName, computedStyle) {
		const body = document.body;
		if (!body) return 0;

		const tempNode = document.createElement(nodeName);
		tempNode.innerHTML = "&nbsp;";
		Object.assign(tempNode.style, {
			fontSize: computedStyle.fontSize,
			fontFamily: computedStyle.fontFamily,
			padding: "0",
			position: "absolute",
		});
		body.appendChild(tempNode);

		// Make sure textarea has only 1 row
		if (tempNode instanceof HTMLTextAreaElement) {
			tempNode.rows = 1;
		}

		// Assume the height of the element is the line-height
		const height = tempNode.offsetHeight;
		body.removeChild(tempNode);

		return height;
	}

	getCursorOffset() {
		const scale = this.getScale();
		const elOffset = this.#calculateElementOffset();
		const elScroll = this.#getElScroll();
		const cursorPosition = this.#getCursorPosition();
		const lineHeight = this.#getLineHeightPx();
		const top = elOffset.top - (elScroll.top * scale) + (cursorPosition.top + lineHeight) * scale;
		const left = elOffset.left - elScroll.left + cursorPosition.left;
		const clientTop = this.el.getBoundingClientRect().top;
		if (this.el.dir !== "rtl") {
			return { top, left, lineHeight, clientTop };
		} else {
			const right = document.documentElement ? document.documentElement.clientWidth - left : 0;
			return { top, right, lineHeight, clientTop };
		}
	}

	#getElScroll() {
		return { top: this.el.scrollTop, left: this.el.scrollLeft };
	}

	#getCursorPosition() {
		return getCaretCoordinates(this.el, this.el.selectionEnd);
	}

	getBeforeCursor() {
		return this.el.selectionStart !== this.el.selectionEnd ? null : this.el.value.substring(0, this.el.selectionEnd);
	}

	getAfterCursor() {
		return this.el.value.substring(this.el.selectionEnd);
	}

	insertAtCursor(value, offset, finalOffset) {
		if (this.el.selectionStart != null) {
			const startPos = this.el.selectionStart;
			const endPos = this.el.selectionEnd;

			// Move selection to beginning of offset
			this.el.selectionStart = this.el.selectionStart + offset;

			// Using execCommand to support undo, but since it's officially 
			// 'deprecated' we need a backup solution, but it won't support undo :(
			let pasted = true;
			try {
				if (!document.execCommand("insertText", false, value)) {
					pasted = false;
				}
			} catch (e) {
				console.error("Error caught during execCommand:", e);
				pasted = false;
			}

			if (!pasted) {
				console.error(
					"execCommand unsuccessful; not supported. Adding text manually, no undo support.");
				textarea.setRangeText(modifiedText, this.el.selectionStart, this.el.selectionEnd, 'end');
			}

			this.el.selectionEnd = this.el.selectionStart = startPos + value.length + offset + (finalOffset ?? 0);
		} else {
			// Using execCommand to support undo, but since it's officially 
			// 'deprecated' we need a backup solution, but it won't support undo :(
			let pasted = true;
			try {
				if (!document.execCommand("insertText", false, value)) {
					pasted = false;
				}
			} catch (e) {
				console.error("Error caught during execCommand:", e);
				pasted = false;
			}

			if (!pasted) {
				console.error(
					"execCommand unsuccessful; not supported. Adding text manually, no undo support.");
				this.el.value += value;
			}
		}
	}
}

/*********************/

/**
 * @typedef {{
 * 	text: string,
 * 	priority?: number,
 * 	info?: Function,
 * 	hint?: string,
 *  showValue?: boolean,
 *  caretOffset?: number
 * }} AutoCompleteEntry
 */
export class TextAreaAutoComplete {
	static globalSeparator = "";
	static enabled = true;
	static insertOnTab = true;
	static insertOnEnter = true;
	static replacer = undefined;
	static lorasEnabled = false;
	static suggestionCount = 20;

	/** @type {Record<string, Record<string, AutoCompleteEntry>>} */
	static groups = {};
	/** @type {Set<string>} */
	static globalGroups = new Set();
	/** @type {Record<string, AutoCompleteEntry>} */
	static globalWords = {};
	/** @type {Record<string, AutoCompleteEntry>} */
	static globalWordsExclLoras = {};

	/** @type {HTMLTextAreaElement} */
	el;

	/** @type {Record<string, AutoCompleteEntry>} */
	overrideWords;
	overrideSeparator = "";

	get words() {
		return this.overrideWords ?? TextAreaAutoComplete.globalWords;
	}

	get separator() {
		return this.overrideSeparator ?? TextAreaAutoComplete.globalSeparator;
	}

	/**
	 * @param {HTMLTextAreaElement} el
	 */
	constructor(el, words = null, separator = null) {
		this.el = el;
		this.helper = new TextAreaCaretHelper(el, () => app.canvas.ds.scale);
		this.dropdown = $el("div.pysssss-autocomplete");
		this.overrideWords = words;
		this.overrideSeparator = separator;

		this.#setup();
	}

	#setup() {
		this.el.addEventListener("keydown", this.#keyDown.bind(this));
		this.el.addEventListener("keypress", this.#keyPress.bind(this));
		this.el.addEventListener("keyup", this.#keyUp.bind(this));
		this.el.addEventListener("click", this.#hide.bind(this));
		this.el.addEventListener("blur", () => setTimeout(() => this.#hide(), 150));
	}

	/**
	 * @param {KeyboardEvent} e
	 */
	#keyDown(e) {
		if (!TextAreaAutoComplete.enabled) return;

		if (this.dropdown.parentElement) {
			// We are visible
			switch (e.key) {
				case "ArrowUp":
					e.preventDefault();
					if (this.selected.index) {
						this.#setSelected(this.currentWords[this.selected.index - 1].wordInfo);
					} else {
						this.#setSelected(this.currentWords[this.currentWords.length - 1].wordInfo);
					}
					break;
				case "ArrowDown":
					e.preventDefault();
					if (this.selected.index === this.currentWords.length - 1) {
						this.#setSelected(this.currentWords[0].wordInfo);
					} else {
						this.#setSelected(this.currentWords[this.selected.index + 1].wordInfo);
					}
					break;
				case "Tab":
					if (TextAreaAutoComplete.insertOnTab) {
						this.#insertItem();
						e.preventDefault();
					}
					break;
			}
		}
	}

	/**
	 * @param {KeyboardEvent} e
	 */
	#keyPress(e) {
		if (!TextAreaAutoComplete.enabled) return;
		if (this.dropdown.parentElement) {
			// We are visible
			switch (e.key) {
				case "Enter":
					if (!e.ctrlKey) {
						if (TextAreaAutoComplete.insertOnEnter) {
							this.#insertItem();
							e.preventDefault();
						}
					}
					break;
			}
		}

		if (!e.defaultPrevented) {
			this.#update();
		}
	}

	#keyUp(e) {
		if (!TextAreaAutoComplete.enabled) return;
		if (this.dropdown.parentElement) {
			// We are visible
			switch (e.key) {
				case "Escape":
					e.preventDefault();
					this.#hide();
					break;
			}
		} else if (e.key.length > 1 && e.key != "Delete" && e.key != "Backspace") {
			return;
		}
		if (!e.defaultPrevented) {
			this.#update();
		}
	}

	#setSelected(item) {
		if (this.selected) {
			this.selected.el.classList.remove("pysssss-autocomplete-item--selected");
		}

		this.selected = item;
		this.selected.el.classList.add("pysssss-autocomplete-item--selected");
	}

	#insertItem() {
		if (!this.selected) return;
		this.selected.el.click();
	}

	#getFilteredWords(term) {
		term = term.toLocaleLowerCase();

		const priorityMatches = [];
		const prefixMatches = [];
		const includesMatches = [];
		for (const word of Object.keys(this.words)) {
			const lowerWord = word.toLocaleLowerCase();
			if (lowerWord === term) {
				// Dont include exact matches
				continue;
			}

			const pos = lowerWord.indexOf(term);
			if (pos === -1) {
				// No match
				continue;
			}

			const wordInfo = this.words[word];
			if (wordInfo.priority) {
				priorityMatches.push({ pos, wordInfo });
			} else if (pos) {
				includesMatches.push({ pos, wordInfo });
			} else {
				prefixMatches.push({ pos, wordInfo });
			}
		}

		priorityMatches.sort(
			(a, b) =>
				b.wordInfo.priority - a.wordInfo.priority ||
				a.wordInfo.text.length - b.wordInfo.text.length ||
				a.wordInfo.text.localeCompare(b.wordInfo.text)
		);

		const top = priorityMatches.length * 0.2;
		return priorityMatches.slice(0, top).concat(prefixMatches, priorityMatches.slice(top), includesMatches).slice(0, TextAreaAutoComplete.suggestionCount);
	}

	#update() {
		let before = this.helper.getBeforeCursor();
		if (before?.length) {
			const m = before.match(/([^,;"|{}()\n]+)$/);
			if (m) {
				before = m[0]
					.replace(/^\s+/, "")
					.replace(/\s/g, "_") || null;
			} else {
				before = null;
			}
		}

		if (!before) {
			this.#hide();
			return;
		}

		this.currentWords = this.#getFilteredWords(before);
		if (!this.currentWords.length) {
			this.#hide();
			return;
		}

		this.dropdown.style.display = "";

		let hasSelected = false;
		const items = this.currentWords.map(({ wordInfo, pos }, i) => {
			const parts = [
				$el("span", {
					textContent: wordInfo.text.substr(0, pos),
				}),
				$el("span.pysssss-autocomplete-highlight", {
					textContent: wordInfo.text.substr(pos, before.length),
				}),
				$el("span", {
					textContent: wordInfo.text.substr(pos + before.length),
				}),
			];

			if (wordInfo.hint) {
				parts.push(
					$el("span.pysssss-autocomplete-pill", {
						textContent: wordInfo.hint,
					})
				);
			}

			if (wordInfo.priority) {
				parts.push(
					$el("span.pysssss-autocomplete-pill", {
						textContent: wordInfo.priority,
					})
				);
			}

			if (wordInfo.value && wordInfo.text !== wordInfo.value && wordInfo.showValue !== false) {
				parts.push(
					$el("span.pysssss-autocomplete-pill", {
						textContent: wordInfo.value,
					})
				);
			}

			if (wordInfo.info) {
				parts.push(
					$el("a.pysssss-autocomplete-item-info", {
						textContent: "ℹ️",
						title: "View info...",
						onclick: (e) => {
							e.stopPropagation();
							wordInfo.info();
							e.preventDefault();
						},
					})
				);
			}
			const item = $el(
				"div.pysssss-autocomplete-item",
				{
				  onclick: () => {
					this.el.focus();
					let value = wordInfo.value ?? wordInfo.text;
					const use_replacer = wordInfo.use_replacer ?? true;
					if (TextAreaAutoComplete.replacer && use_replacer) {
					  value = TextAreaAutoComplete.replacer(value);
					}
					value = this.#escapeParentheses(value);
					
					const afterCursor = this.helper.getAfterCursor();
					const shouldAddSeparator = !afterCursor.trim().startsWith(this.separator.trim());
					this.helper.insertAtCursor(
					  value + (shouldAddSeparator ? this.separator : ''),
					  -before.length,
					  wordInfo.caretOffset
					);			
					setTimeout(() => {
					  this.#update();
					}, 150);
				  },
				},
				parts
			  );

			if (wordInfo === this.selected) {
				hasSelected = true;
			}

			wordInfo.index = i;
			wordInfo.el = item;

			return item;
		});

		this.#setSelected(hasSelected ? this.selected : this.currentWords[0].wordInfo);
		this.dropdown.replaceChildren(...items);

		if (!this.dropdown.parentElement) {
			document.body.append(this.dropdown);
		}

		const position = this.helper.getCursorOffset();
		this.dropdown.style.left = (position.left ?? 0) + "px";
		this.dropdown.style.top = (position.top ?? 0) + "px";
		this.dropdown.style.maxHeight = (window.innerHeight - position.top) + "px";
	}

	#escapeParentheses(text) {
		return text.replace(/\(/g, '\\(').replace(/\)/g, '\\)');
	  }

	#hide() {
		this.selected = null;
		this.dropdown.remove();
	}

	static updateWords(id, words, addGlobal = true) {
		const isUpdate = id in TextAreaAutoComplete.groups;
		TextAreaAutoComplete.groups[id] = words;
		if (addGlobal) {
			TextAreaAutoComplete.globalGroups.add(id);
		}

		if (isUpdate) {
			// Remerge all words
			TextAreaAutoComplete.globalWords = Object.assign(
				{},
				...Object.keys(TextAreaAutoComplete.groups)
					.filter((k) => TextAreaAutoComplete.globalGroups.has(k))
					.map((k) => TextAreaAutoComplete.groups[k])
			);
		} else if (addGlobal) {
			// Just insert the new words
			Object.assign(TextAreaAutoComplete.globalWords, words);
		}
	}
}
