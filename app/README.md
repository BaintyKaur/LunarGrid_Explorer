# LunarGrid Explorer 3D

A 3D globe of the Moon. Drag to rotate, scroll to zoom and click a place to see crater density, hotspot tests,
terrain and sunlight for the three crater catalogues.

Built with [Three.js](https://threejs.org) and [Vite](https://vitejs.dev). The data is a single file,
`public/dashboard_data.json`, made by the analysis scripts in `../scripts`.

## Run it on your computer

```
cd app
npm install
npm run dev
```

Open the address that `npm run dev` prints (normally http://localhost:5173).

## Build for the web

```
npm run build
```

The finished site is written to `app/dist`. Pushing to the `main` branch builds and publishes it automatically with
GitHub Actions (see `.github/workflows/deploy.yml`).

## How it works

- `src/model.js` loads the data, calculates values and colours, and writes the statistics panel.
- `src/globe.js` draws the Moon as a sphere painted from the data, and handles clicks, the marker and the camera.
- `src/main.js` connects the controls, the globe and the panel.
