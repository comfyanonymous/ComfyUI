/** @type {import('jest').Config} */
const config = {
	testEnvironment: "jsdom",
	setupFiles: ["./globalSetup.js"],
	clearMocks: true,
	resetModules: true,
};

module.exports = config;
