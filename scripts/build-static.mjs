import {cp, mkdir, rm} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import path from "node:path";
import {vendorBrowser} from "./vendor-browser.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const output = path.join(root, "dist/browser");
await vendorBrowser();
await rm(output, {recursive: true, force: true});
await mkdir(output, {recursive: true});
await cp(path.join(root, "morpholab/static"), output, {recursive: true});
console.log("Static site built in dist/browser. Serve this directory over HTTPS (or localhost).");
