'use strict';

var process = require('process');

var tf;
if (process.env.WEBPPL_TFJS_NODE === '1') {
  tf = require('@tensorflow/tfjs-node');
}
else {
  tf = require('@tensorflow/tfjs-core');
}

module.exports = tf;
