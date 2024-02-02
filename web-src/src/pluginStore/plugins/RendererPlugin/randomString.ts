export default function randomString(length: number) {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

    text = possible.charAt(Math.floor(Math.random() * 52));
    for (let i = 0; i < length - 1; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
