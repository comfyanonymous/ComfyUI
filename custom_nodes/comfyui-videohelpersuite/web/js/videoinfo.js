import { app } from '../../../scripts/app.js'


function getVideoMetadata(file) {
    return new Promise((r) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const videoData = new Uint8Array(event.target.result);
            const dataView = new DataView(videoData.buffer);

            let decoder = new TextDecoder();
            // Check for known valid magic strings
            if (dataView.getUint32(0) == 0x1A45DFA3) {
                //webm/mkv (both use EBML/Matroska format)
                //see http://wiki.webmproject.org/webm-metadata/global-metadata
                //and https://www.matroska.org/technical/elements.html
                //contrary to specs, tag seems consistently at start
                //COMMENT + 0x4487 + packed length?
                //length 0x8d8 becomes 0x48d8
                //
                //description for variable length ints https://github.com/ietf-wg-cellar/ebml-specification/blob/master/specification.markdown
                let offset = 4 + 8; //COMMENT is 7 chars + 1 to realign
                while(offset < videoData.length-16) {
                    //Check for text tags
                    if (dataView.getUint16(offset) == 0x4487) {
                        //check that name of tag is COMMENT
                        const name = String.fromCharCode(...videoData.slice(offset-7,offset));
                        if (name === "COMMENT") {
                            let vint = dataView.getUint32(offset+2);
                            let n_octets = Math.clz32(vint)+1;
                            if (n_octets < 4) {//250MB sanity cutoff
                                let length = (vint >> (8*(4-n_octets))) & ~(1 << (7*n_octets));
                                const content = decoder.decode(videoData.slice(offset+2+n_octets, offset+2+n_octets+length));
                                let json = JSON.parse(content);
                                r(json);
                                return;
                            }
                        }
                    }
                    offset+=1;
                }
            } else if (dataView.getUint32(4) == 0x66747970 && dataView.getUint32(8) == 0x69736F6D) {
                //mp4
                //see https://developer.apple.com/documentation/quicktime-file-format
                //Seems to make no guarantee for alignment
                let offset = videoData.length-4;
                while (offset > 16) {//rough safe guess
                    if (dataView.getUint32(offset) == 0x64617461) {//any data tag
                        if (dataView.getUint32(offset - 8) == 0xa9636d74) {//cmt data tag
                            let type = dataView.getUint32(offset+4); //seemingly 1
                            let locale = dataView.getUint32(offset+8); //seemingly 0
                            let size = dataView.getUint32(offset-4) - 4*4;
                            const content = decoder.decode(videoData.slice(offset+12, offset+12+size));
                            const json = JSON.parse(content);
                            r(json);
                            return;
                        }
                    }

                    offset-=1;
                }
            } else {
                console.error("Unknown magic: " + dataView.getUint32(0))
            }
            r();
            return;
        };

        reader.readAsArrayBuffer(file);
    });
}
function isVideoFile(file) {
    if (file?.name?.endsWith(".webm")) {
        return true;
    }
    if (file?.name?.endsWith(".mp4")) {
        return true;
    }
    if (file?.name?.endsWith(".mkv")) {
        return true;
    }

    return false;
}

let originalHandleFile = app.handleFile;
app.handleFile = handleFile;
let fileInput = document.getElementById("comfy-file-input")
//hijack comfy-file-input to allow webm/mp4/mkv
fileInput.accept += ",video/webm,video/mp4,video/x-matroska";

async function handleFile(file) {
    if (file?.type?.startsWith("video/") || isVideoFile(file)) {
        const videoInfo = await getVideoMetadata(file);
        if (videoInfo?.workflow) {
            await app.loadGraphData(videoInfo.workflow);
            return
        }
    }
    return await originalHandleFile.apply(this, arguments);
}
