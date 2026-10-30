'use strict';

const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync('public/admin.html', 'utf8');
const match = html.match(/<script>([\s\S]*?)<\/script>/i);

if (!match) {
  throw new Error('No inline admin script was found in public/admin.html');
}

new vm.Script(match[1], { filename: 'public/admin.html<script>' });
process.stdout.write('Admin dashboard script syntax OK\n');
