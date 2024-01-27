// Simple date formatter
const parts = {
    d: (d: Date) => d.getDate(),
    M: (d: Date) => d.getMonth() + 1,
    h: (d: Date) => d.getHours(),
    m: (d: Date) => d.getMinutes(),
    s: (d: Date) => d.getSeconds(),
};
const format =
    Object.keys(parts)
        .map(k => k + k + '?')
        .join('|') + '|yyy?y?';

export function formatDate(text: string, date: Date) {
    return text.replace(new RegExp(format, 'g'), function (text) {
        if (text === 'yy') return (date.getFullYear() + '').substring(2);
        if (text === 'yyyy') return date.getFullYear().toString();
        if (text[0] in parts) {
            const p = parts[text[0] as keyof typeof parts](date);
            return (p + '').padStart(text.length, '0');
        }

        return text;
    });
}

export async function addStylesheet(urlOrFile: string, relativeTo: string) {
    return new Promise((res, rej) => {
        let url;
        if (urlOrFile.endsWith('.js')) {
            url = urlOrFile.substr(0, urlOrFile.length - 2) + 'css';
        } else {
            url = new URL(urlOrFile, relativeTo ?? `${window.location.protocol}//${window.location.host}`).toString();
        }
        $el('link', {
            parent: document.head,
            rel: 'stylesheet',
            type: 'text/css',
            href: url,
            onload: res,
            onerror: rej,
        });
    });
}

export function sanitizeNodeName(string: string) {
    let entityMap = {
        '&': '',
        '<': '',
        '>': '',
        '"': '',
        "'": '',
        '`': '',
        '=': '',
    };

    return String(string).replace(/[&<>"'`=]/g, function fromEntityMap(s) {
        return entityMap[s as keyof typeof entityMap];
    });
}

type ElementProps = {
    width?: number;
    height?: number;
    parent?: Element;
    $?: (el: Element) => void;
    title?: string;
    dataset?: DOMStringMap;
    style?: { [key: string]: string };
    textContent?: string | null;
    onclick?: () => void;
    for?: string;
    accept?: string;
    onchange?: (v: any) => void;
    innerHTML?: string;
    href?: string;
    download?: string | null;
    min?: string;
    max?: string;
    value?: any;
    checked?: boolean;
    rel?: string;
    onload?: (value: any) => void;
    onerror?: (reason?: any) => void;
    oninput?: (
        i: InputEvent & {
            target?: { value: any };
            srcElement?: { value: any };
        }
    ) => void;
};

/** tag is an HTML Element Tag and optional classes e.g. div.class1.class2 */
export function $el(
    tag: string,
    propsOrChildren?: string | Element | Element[] | ElementProps,
    children?: Element[]
): Element {
    const split = tag.split('.');
    const element = document.createElement(split.shift()!);
    if (split.length > 0) {
        element.classList.add(...split);
    }

    if (propsOrChildren) {
        if (typeof propsOrChildren === 'string') {
            propsOrChildren = { textContent: propsOrChildren };
        } else if (propsOrChildren instanceof Element) {
            propsOrChildren = [propsOrChildren];
        }
        if (Array.isArray(propsOrChildren)) {
            element.append(...propsOrChildren);
        } else {
            if (propsOrChildren) {
                const { parent, $: cb, dataset, style } = propsOrChildren as ElementProps;
                delete (propsOrChildren as ElementProps).parent;
                delete (propsOrChildren as ElementProps).$;
                delete (propsOrChildren as ElementProps).dataset;
                delete (propsOrChildren as ElementProps).style;

                if (Object.hasOwn(propsOrChildren as ElementProps, 'for')) {
                    element.setAttribute('for', (propsOrChildren as ElementProps).for!);
                }

                if (style) {
                    Object.assign(element.style, style);
                }

                if (dataset) {
                    Object.assign(element.dataset, dataset);
                }

                Object.assign(element, propsOrChildren);
                if (children) {
                    element.append(...(children instanceof Array ? children : [children]));
                }

                if (parent) {
                    parent.append(element);
                }

                if (cb) {
                    cb(element);
                }
            }
        }
    }
    return element;
}
