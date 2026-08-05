# UAII documentation site

Static Next.js export for Universal AI Inference Runtime docs.

## Develop

```bash
cd website
npm ci
npm run dev
```

## Build

```bash
npm run build
# output in out/
npm start   # serves out/ via serve
```

## Theme

Light / dark themes are client-controlled (`localStorage` key `uaii-theme`). Dark defaults to a Cursor-inspired matte black / gray / white palette; light uses cool paper and charcoal ink. Toggle lives in the site nav.
