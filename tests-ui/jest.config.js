/** @type {import('jest').Config} */
const config = {
	testEnvironment: "jsdom",
	setupFiles: ["./globalSetup.js"],
	setupFilesAfterEnv: ["./afterSetup.js"],
	clearMocks: true,
	resetModules: true,
	testTimeout: 10000
};

module.exports = config;
