interface Setting {
    id: string;
    name: string;
    type: string;
    defaultValue?: any;
    onChange?: () => any;
    attrs?: Record<string, any>;
}

export const settings: Setting[] = [
    {
        id: 'Comfy.ConfirmClear',
        name: 'Require confirmation when clearing workflow',
        type: 'boolean',
        defaultValue: true,
        onChange: () => undefined,
    },
    {
        id: 'Comfy.PromptFilename',
        name: 'Prompt for filename when saving workflow',
        type: 'boolean',
        defaultValue: true,
        onChange: () => undefined,
    },
    {
        id: 'Comfy.PreviewFormat',
        name: 'When displaying a preview in the image widget, convert it to a lightweight image, e.g. webp, jpeg, webp;50, etc.',
        type: 'text',
        defaultValue: '',
        onChange: () => undefined,
    },
    {
        id: 'Comfy.DisableSliders',
        name: 'Disable sliders.',
        type: 'boolean',
        defaultValue: false,
        onChange: () => undefined,
    },
    {
        id: 'Comfy.DisableFloatRounding',
        name: 'Disable rounding floats (requires page reload).',
        type: 'boolean',
        defaultValue: false,
        onChange: () => undefined,
    },
    {
        id: 'Comfy.FloatRoundingPrecision',
        name: 'Decimal places [0 = auto] (requires page reload).',
        type: 'slider',
        attrs: {
            min: 0,
            max: 6,
            step: 1,
        },
        defaultValue: 0,
        onChange: () => undefined,
    },
];
