const path = require("path");
/** @type {import('jest').Config} */
const config = {
	testEnvironment: "jsdom",
	// transform: {
	// 	"^.+\\.[t|j]sx?$": "babel-jest",
	// },
	setupFiles: ["./globalSetup.js"],
	// moduleDirectories: ["node_modules", path.resolve("../web/scripts")],
	// moduleNameMapper: {
	// 	"./api.js": path.resolve("../web/scripts/api.js"),
	// 	"./api": path.resolve("../web/scripts/api.js"),
	// },
};

module.exports = config;
