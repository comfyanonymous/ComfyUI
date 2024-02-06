import {
    BooleanInputProps,
    ComboInputProps,
    NumberInputProps,
    SliderInputProps,
    TextInputProps,
} from '../types/many.ts';

export function BooleanInput({ id, value, onChange, setSettingValue }: BooleanInputProps) {
    return (
        <input
            id={id}
            type="checkbox"
            checked={value}
            onChange={event => {
                const isChecked = event.target.checked;
                if (onChange !== undefined) {
                    onChange(isChecked);
                }
                setSettingValue(id, isChecked);
            }}
        />
    );
}

export function NumberInput({ id, value, setter, attrs }: NumberInputProps) {
    return (
        <input
            id={id}
            type="number"
            value={value}
            onInput={e => {
                const target = e.target as HTMLInputElement;
                setter(target.value);
            }}
            {...attrs}
        />
    );
}

export function SliderInput({ id, value, setter, attrs }: SliderInputProps) {
    return (
        <div style={{ display: 'grid', gridAutoFlow: 'column' }}>
            <input
                {...attrs}
                value={value}
                type="range"
                onInput={e => {
                    const target = e.target as HTMLInputElement;
                    setter(target.value);
                    if (target.nextElementSibling instanceof HTMLInputElement) {
                        target.nextElementSibling.value = target.value;
                    }
                }}
            />
            <input
                id={id}
                type="number"
                value={value}
                style={{ maxWidth: '4rem' }}
                onInput={e => {
                    const target = e.target as HTMLInputElement;
                    setter(target.value);
                    if (target.previousElementSibling instanceof HTMLInputElement) {
                        target.previousElementSibling.value = target.value;
                    }
                }}
                {...attrs}
            />
        </div>
    );
}

export function ComboInput({ options, setter, value }: ComboInputProps) {
    return (
        <select
            onInput={e => {
                const target = e.target as HTMLSelectElement;
                setter(target.value);
            }}
        >
            {(typeof options === 'function' ? options(value) : options || []).map(opt => {
                if (typeof opt === 'string') {
                    opt = { text: opt };
                }
                const v = opt.value ?? opt.text;
                return (
                    <option value={v} selected={value + '' === v + ''}>
                        {opt.text}
                    </option>
                );
            })}
        </select>
    );
}

export function TextInput({ id, value, setter, attrs }: TextInputProps) {
    return (
        <input
            value={value}
            id={id}
            onInput={e => {
                const target = e.target as HTMLInputElement;
                setter(target.value);
            }}
            {...attrs}
        />
    );
}
