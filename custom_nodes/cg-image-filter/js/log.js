import { app } from "../../scripts/app.js";

export class Log {
    static log(s) { if (s) console.log(s) }
    static error(e) { console.error(e) }
    static detail(s) {
        if (app.ui.settings.getSettingValue("Image Filter.Z.Detailed Logging")) Log.log(s)
    }
    static message_in(message, extra) {
        if (!app.ui.settings.getSettingValue("Image Filter.Z.Detailed Logging")) return
        if (message.detail && !message.detail.tick) Log.log(`--> ${JSON.stringify(message.detail)}` + (extra ? ` ${extra}` : ""))
        if (message.detail && message.detail.tick) Log.log(`--> tick`)
    }
    static message_out(response, extra) {
        if (!app.ui.settings.getSettingValue("Image Filter.Z.Detailed Logging")) return
        Log.log(`"<-- ${JSON.stringify(response)}` + (extra ? ` ${extra}` : ""))
    }
}