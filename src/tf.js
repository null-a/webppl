'use strict';

var process = require('process');

var tf = require('@tensorflow/tfjs-core');

if (process.env.WEBPPL_TFJS_NODE === '1') {
  require('@tensorflow/tfjs-node');
}

module.exports = tf;
