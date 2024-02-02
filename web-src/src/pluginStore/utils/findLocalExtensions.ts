// Finds all .js files in the /extensions directory

import fs from 'fs/promises';
import path from 'path';

export const getLocalExtensions = async () => {
    const extensionsDir = path.join(__dirname, '..', 'extensions');
    return findJsFiles(extensionsDir);
};

async function findJsFiles(dir: string): Promise<string[]> {
    const files = await fs.readdir(dir, { withFileTypes: true });
    let jsFiles: string[] = [];

    for (const file of files) {
        const fullPath = path.join(dir, file.name);
        if (file.isDirectory()) {
            jsFiles = jsFiles.concat(await findJsFiles(fullPath));
        } else if (file.isFile() && file.name.endsWith('.js')) {
            jsFiles.push(fullPath);
        }
    }

    return jsFiles;
}
