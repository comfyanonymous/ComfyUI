// @ts-check
/// <reference path="../node_modules/@types/jest/index.d.ts" />
const { start } = require("../utils");
const lg = require("../utils/litegraph");

describe("users", () => {
	beforeEach(() => {
		lg.setup(global);
	});

	afterEach(() => {
		lg.teardown(global);
	});

	function expectNoUserScreen() {
		// Ensure login isnt visible
		const selection = document.querySelectorAll("#comfy-user-selection")?.[0];
		expect(selection["style"].display).toBe("none");
		const menu = document.querySelectorAll(".comfy-menu")?.[0];
		expect(window.getComputedStyle(menu)?.display).not.toBe("none");
	}

	describe("multi-user", () => {
		function mockAddStylesheet() {
			const utils = require("../../web/scripts/utils");
			utils.addStylesheet = jest.fn().mockReturnValue(Promise.resolve());
		}

		async function waitForUserScreenShow() {
			mockAddStylesheet();

			// Wait for "show" to be called
			const { UserSelectionScreen } = require("../../web/scripts/ui/userSelection");
			let resolve, reject;
			const fn = UserSelectionScreen.prototype.show;
			const p = new Promise((res, rej) => {
				resolve = res;
				reject = rej;
			});
			jest.spyOn(UserSelectionScreen.prototype, "show").mockImplementation(async (...args) => {
				const res = fn(...args);
				await new Promise(process.nextTick); // wait for promises to resolve
				resolve();
				return res;
			});
			// @ts-ignore
			setTimeout(() => reject("timeout waiting for UserSelectionScreen to be shown."), 500);
			await p;
			await new Promise(process.nextTick); // wait for promises to resolve
		}

		async function testUserScreen(onShown, users) {
			if (!users) {
				users = {};
			}
			const starting = start({
				resetEnv: true,
				userConfig: { storage: "server", users },
			});

			// Ensure no current user
			expect(localStorage["Comfy.userId"]).toBeFalsy();
			expect(localStorage["Comfy.userName"]).toBeFalsy();

			await waitForUserScreenShow();

			const selection = document.querySelectorAll("#comfy-user-selection")?.[0];
			expect(selection).toBeTruthy();

			// Ensure login is visible
			expect(window.getComputedStyle(selection)?.display).not.toBe("none");
			// Ensure menu is hidden
			const menu = document.querySelectorAll(".comfy-menu")?.[0];
			expect(window.getComputedStyle(menu)?.display).toBe("none");

			const isCreate = await onShown(selection);

			// Submit form
			selection.querySelectorAll("form")[0].submit();
			await new Promise(process.nextTick); // wait for promises to resolve

			// Wait for start
			const s = await starting;

			// Ensure login is removed
			expect(document.querySelectorAll("#comfy-user-selection")).toHaveLength(0);
			expect(window.getComputedStyle(menu)?.display).not.toBe("none");

			// Ensure settings + templates are saved
			const { api } = require("../../web/scripts/api");
			expect(api.createUser).toHaveBeenCalledTimes(+isCreate);
			expect(api.storeSettings).toHaveBeenCalledTimes(+isCreate);
			expect(api.storeUserData).toHaveBeenCalledTimes(+isCreate);
			if (isCreate) {
				expect(api.storeUserData).toHaveBeenCalledWith("comfy.templates.json", null, { stringify: false });
				expect(s.app.isNewUserSession).toBeTruthy();
			} else {
				expect(s.app.isNewUserSession).toBeFalsy();
			}

			return { users, selection, ...s };
		}

		it("allows user creation if no users", async () => {
			const { users } = await testUserScreen((selection) => {
				// Ensure we have no users flag added
				expect(selection.classList.contains("no-users")).toBeTruthy();

				// Enter a username
				const input = selection.getElementsByTagName("input")[0];
				input.focus();
				input.value = "Test User";

				return true;
			});

			expect(users).toStrictEqual({
				"Test User!": "Test User",
			});

			expect(localStorage["Comfy.userId"]).toBe("Test User!");
			expect(localStorage["Comfy.userName"]).toBe("Test User");
		});
		it("allows user creation if no current user but other users", async () => {
			const users = {
				"Test User 2!": "Test User 2",
			};

			await testUserScreen((selection) => {
				expect(selection.classList.contains("no-users")).toBeFalsy();

				// Enter a username
				const input = selection.getElementsByTagName("input")[0];
				input.focus();
				input.value = "Test User 3";
				return true;
			}, users);

			expect(users).toStrictEqual({
				"Test User 2!": "Test User 2",
				"Test User 3!": "Test User 3",
			});

			expect(localStorage["Comfy.userId"]).toBe("Test User 3!");
			expect(localStorage["Comfy.userName"]).toBe("Test User 3");
		});
		it("allows user selection if no current user but other users", async () => {
			const users = {
				"A!": "A",
				"B!": "B",
				"C!": "C",
			};

			await testUserScreen((selection) => {
				expect(selection.classList.contains("no-users")).toBeFalsy();

				// Check user list
				const select = selection.getElementsByTagName("select")[0];
				const options = select.getElementsByTagName("option");
				expect(
					[...options]
						.filter((o) => !o.disabled)
						.reduce((p, n) => {
							p[n.getAttribute("value")] = n.textContent;
							return p;
						}, {})
				).toStrictEqual(users);

				// Select an option
				select.focus();
				select.value = options[2].value;

				return false;
			}, users);

			expect(users).toStrictEqual(users);

			expect(localStorage["Comfy.userId"]).toBe("B!");
			expect(localStorage["Comfy.userName"]).toBe("B");
		});
		it("doesnt show user screen if current user", async () => {
			const starting = start({
				resetEnv: true,
				userConfig: {
					storage: "server",
					users: {
						"User!": "User",
					},
				},
				localStorage: {
					"Comfy.userId": "User!",
					"Comfy.userName": "User",
				},
			});
			await new Promise(process.nextTick); // wait for promises to resolve

			expectNoUserScreen();

			await starting;
		});
		it("allows user switching", async () => {
			const { app } = await start({
				resetEnv: true,
				userConfig: {
					storage: "server",
					users: {
						"User!": "User",
					},
				},
				localStorage: {
					"Comfy.userId": "User!",
					"Comfy.userName": "User",
				},
			});

			// cant actually test switching user easily but can check the setting is present
			expect(app.ui.settings.settingsLookup["Comfy.SwitchUser"]).toBeTruthy();
		});
	});
	describe("single-user", () => {
		it("doesnt show user creation if no default user", async () => {
			const { app } = await start({
				resetEnv: true,
				userConfig: { migrated: false, storage: "server" },
			});
			expectNoUserScreen();

			// It should store the settings
			const { api } = require("../../web/scripts/api");
			expect(api.storeSettings).toHaveBeenCalledTimes(1);
			expect(api.storeUserData).toHaveBeenCalledTimes(1);
			expect(api.storeUserData).toHaveBeenCalledWith("comfy.templates.json", null, { stringify: false });
			expect(app.isNewUserSession).toBeTruthy();
		});
		it("doesnt show user creation if default user", async () => {
			const { app } = await start({
				resetEnv: true,
				userConfig: { migrated: true, storage: "server" },
			});
			expectNoUserScreen();

			// It should store the settings
			const { api } = require("../../web/scripts/api");
			expect(api.storeSettings).toHaveBeenCalledTimes(0);
			expect(api.storeUserData).toHaveBeenCalledTimes(0);
			expect(app.isNewUserSession).toBeFalsy();
		});
		it("doesnt allow user switching", async () => {
			const { app } = await start({
				resetEnv: true,
				userConfig: { migrated: true, storage: "server" },
			});
			expectNoUserScreen();

			expect(app.ui.settings.settingsLookup["Comfy.SwitchUser"]).toBeFalsy();
		});
	});
	describe("browser-user", () => {
		it("doesnt show user creation if no default user", async () => {
			const { app } = await start({
				resetEnv: true,
				userConfig: { migrated: false, storage: "browser" },
			});
			expectNoUserScreen();

			// It should store the settings
			const { api } = require("../../web/scripts/api");
			expect(api.storeSettings).toHaveBeenCalledTimes(0);
			expect(api.storeUserData).toHaveBeenCalledTimes(0);
			expect(app.isNewUserSession).toBeFalsy();
		});
		it("doesnt show user creation if default user", async () => {
			const { app } = await start({
				resetEnv: true,
				userConfig: { migrated: true, storage: "server" },
			});
			expectNoUserScreen();

			// It should store the settings
			const { api } = require("../../web/scripts/api");
			expect(api.storeSettings).toHaveBeenCalledTimes(0);
			expect(api.storeUserData).toHaveBeenCalledTimes(0);
			expect(app.isNewUserSession).toBeFalsy();
		});
		it("doesnt allow user switching", async () => {
			const { app } = await start({
				resetEnv: true,
				userConfig: { migrated: true, storage: "browser" },
			});
			expectNoUserScreen();

			expect(app.ui.settings.settingsLookup["Comfy.SwitchUser"]).toBeFalsy();
		});
	});
});
