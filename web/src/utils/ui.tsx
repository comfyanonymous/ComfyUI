import React, { RefObject, useEffect, useState } from 'react';

type RawInput = {
    text: string;
    value?: string;
    tooltip?: string;
    selected?: boolean;
};

interface RadioInputProps {
    index: number;
    item: RawInput;
    handleSelect: (index: number) => void;
}

export function toggleSwitch(
    name: string,
    items: (RawInput | string)[],
    { onChange, ref }: { ref?: RefObject<HTMLDivElement>; onChange?: (value: any) => void } = {}
) {
    let selectedIndex: number | null = null;

    // TODO: if none is selected, select the first one

    const RadioInput = ({ index, item, handleSelect }: RadioInputProps) => {
        const [selected, setSelected] = useState(false);

        useEffect(() => {
            if (item.selected) {
                setSelected(item.selected);
                handleSelect(index);
            }
        }, []);

        return (
            <label className={selected ? 'comfy-toggle-selected' : ''} title={item.tooltip ?? ''}>
                <input
                    name={name}
                    type="radio"
                    value={item.value ?? item.text}
                    checked={(item as RawInput).selected}
                    onChange={() => {
                        setSelected(selected => !selected);
                        handleSelect(index);
                    }}
                />
            </label>
        );
    };

    const handleSelected = (index: number) => {
        onChange?.({
            item: items[index],
            prev: selectedIndex == null ? undefined : items[selectedIndex],
        });

        selectedIndex = index;
    };

    const elements = items.map((item, i) => {
        if (typeof item === 'string') {
            item = { text: item };
        }

        if (!item.value) {
            item.value = item.text;
        }

        return <RadioInput key={i} handleSelect={handleSelected} item={item} index={i} />;
    });

    return (
        <div ref={ref} className="comfy-toggle-switch" style={{ display: 'none' }}>
            {elements}
        </div>
    );
}

export function dragElement(dragEl: HTMLDivElement, addSetting: any) {
    var posDiffX = 0,
        posDiffY = 0,
        posStartX = 0,
        posStartY = 0,
        newPosX = 0,
        newPosY = 0;

    if (dragEl.getElementsByClassName('drag-handle')[0]) {
        // if present, the handle is where you move the DIV from:
        (dragEl.getElementsByClassName('drag-handle')[0] as HTMLElement).onmousedown = dragMouseDown;
    } else {
        // otherwise, move the DIV from anywhere inside the DIV:
        dragEl.onmousedown = dragMouseDown;
    }

    // When the element resizes (e.g. view queue) ensure it is still in the windows bounds
    const resizeObserver = new ResizeObserver(() => ensureInBounds());
    resizeObserver.observe(dragEl);

    function ensureInBounds() {
        if (dragEl.classList.contains('comfy-menu-manual-pos')) {
            newPosX = Math.min(document.body.clientWidth - dragEl.clientWidth, Math.max(0, dragEl.offsetLeft));
            newPosY = Math.min(document.body.clientHeight - dragEl.clientHeight, Math.max(0, dragEl.offsetTop));

            positionElement();
        }
    }

    function positionElement() {
        const halfWidth = document.body.clientWidth / 2;
        const anchorRight = newPosX + dragEl.clientWidth / 2 > halfWidth;

        // set the element's new position:
        if (anchorRight) {
            dragEl.style.left = 'unset';
            dragEl.style.right = document.body.clientWidth - newPosX - dragEl.clientWidth + 'px';
        } else {
            dragEl.style.left = newPosX + 'px';
            dragEl.style.right = 'unset';
        }

        dragEl.style.top = newPosY + 'px';
        dragEl.style.bottom = 'unset';

        if (savePos) {
            localStorage.setItem(
                'Comfy.MenuPosition',
                JSON.stringify({
                    x: dragEl.offsetLeft,
                    y: dragEl.offsetTop,
                })
            );
        }
    }

    function restorePos() {
        let pos = localStorage.getItem('Comfy.MenuPosition');
        if (pos) {
            const newPos = JSON.parse(pos);
            newPosX = newPos.x;
            newPosY = newPos.y;
            positionElement();
            ensureInBounds();
        }
    }

    let savePos: undefined | any = undefined;
    addSetting({
        id: 'Comfy.MenuPosition',
        name: 'Save menu position',
        type: 'boolean',
        defaultValue: savePos,
        onChange(value: any) {
            if (savePos === undefined && value) {
                restorePos();
            }
            savePos = value;
        },
    });

    function dragMouseDown(e: MouseEvent) {
        e = e || window.event;
        e.preventDefault();
        // get the mouse cursor position at startup:
        posStartX = e.clientX;
        posStartY = e.clientY;
        document.onmouseup = closeDragElement;
        // call a function whenever the cursor moves:
        document.onmousemove = elementDrag;
    }

    function elementDrag(e: MouseEvent) {
        e = e || window.event;
        e.preventDefault();

        dragEl.classList.add('comfy-menu-manual-pos');

        // calculate the new cursor position:
        posDiffX = e.clientX - posStartX;
        posDiffY = e.clientY - posStartY;
        posStartX = e.clientX;
        posStartY = e.clientY;

        newPosX = Math.min(document.body.clientWidth - dragEl.clientWidth, Math.max(0, dragEl.offsetLeft + posDiffX));
        newPosY = Math.min(document.body.clientHeight - dragEl.clientHeight, Math.max(0, dragEl.offsetTop + posDiffY));

        positionElement();
    }

    window.addEventListener('resize', () => {
        ensureInBounds();
    });

    function closeDragElement() {
        // stop moving when mouse button is released:
        document.onmouseup = null;
        document.onmousemove = null;
    }
}
