import {$el} from "./utils.ts";
import {IComfyUserSettings} from "../types/interfaces.ts";

export class ComfyUserSettings implements IComfyUserSettings {
    isNewUserSession: boolean | null;
    storageLocation: string | null;
    multiUserServer: boolean | null;

    constructor() {
        this.storageLocation = null;
        this.multiUserServer = null;
        this.isNewUserSession = null;
    }

    async #migrateSettings() {
        this.isNewUserSession = true;

        // Store all current settings
        const settings = Object.keys(this.ui.settings).reduce((p: { [x: string]: any }, n) => {
            const v = localStorage[`Comfy.Settings.${n}`];
            if (v) {
                try {
                    p[n] = JSON.parse(v);
                } catch (error) {
                }
            }
            return p;
        }, {});

        await this.api.storeSettings(settings);
    }

    async setUser() {
        const userConfig = await this.api.getUserConfig();
        this.storageLocation = userConfig.storage;
        if (typeof userConfig.migrated == 'boolean') {
            // Single user mode migrated true/false for if the default user is created
            if (!userConfig.migrated && this.storageLocation === 'server') {
                // Default user not created yet
                await this.#migrateSettings();
            }
            return;
        }
        this.multiUserServer = true;
        let user = localStorage['Comfy.userId'];
        const users = userConfig.users ?? {};
        if (!user || !users[user]) {
            // This will rarely be hit so move the loading to on demand
            const {UserSelectionScreen} = await import('./ui/userSelection');
            this.ui.menuContainer.style.display = 'none';
            const {userId, username, created} = await new UserSelectionScreen().show(users, user);
            this.ui.menuContainer.style.display = '';
            user = userId;
            localStorage['Comfy.userName'] = username;
            localStorage['Comfy.userId'] = user;
            if (created) {
                this.api.user = user;
                await this.#migrateSettings();
            }
        }
        this.api.user = user;
        this.ui.settings.addSetting({
            id: 'Comfy.SwitchUser',
            name: 'Switch User',
            defaultValue: 'any',
            type: (name: string) => {
                let currentUser = localStorage['Comfy.userName'];
                if (currentUser) {
                    currentUser = ` (${currentUser})`;
                }
                return $el('tr', [
                    $el('td', [
                        $el('label', {
                            textContent: name,
                        }),
                    ]),
                    $el('td', [
                        $el('button', {
                            textContent: name + (currentUser ?? ''),
                            onclick: () => {
                                delete localStorage['Comfy.userId'];
                                delete localStorage['Comfy.userName'];
                                window.location.reload();
                            },
                        }),
                    ]),
                ]);
            },
        });
    }
}
