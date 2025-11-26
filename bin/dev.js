#!/usr/bin/env node
import('../dist/cli.js').then(({ default: run }) => run(['dev', ...process.argv.slice(2)]));
