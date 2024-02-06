export class Event {
    // target: Layer;
    name: string;
    // payload: any;
    // currentTarget: Layer | undefined;
    private _propagate: boolean;
    private _defaults: boolean;
    constructor(name: string) {
        this.name = name;
        // this.target = target;
        // this.currentTarget = currentTarget;
        this._propagate = true;
        this._defaults = true;
    }
    get propagate() {
        return this._propagate;
    }
    get defaults() {
        return this._defaults;
    }
    stopPropagation() {
        this._propagate = false;
    }
    preventDefault() {
        this._defaults = false;
    }
}
