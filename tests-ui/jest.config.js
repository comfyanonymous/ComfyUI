/** @type {import('jest').Config} */
const config = {
	testEnvironment: "jsdom",
	setupFiles: ["./globalSetup.js"],
};

module.exports = config;
