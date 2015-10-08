'use strict';

// ........

var _ = require('underscore');
var fs = require('fs');
var webppl = require('../src/main.js');
var erp = require('../src/erp');
var assert = require('assert');
var https = require('https');
var querystring = require('querystring');

var examplesDir = './examples/';

var examples = [
  'lda',
  'hmmIncremental'
];

var loadExample = function(example) {
  var filename = examplesDir + example + '.wppl';
  return fs.readFileSync(filename, 'utf-8');
};

var estimatePI = function(n) {
  var loop = function(i, samples) {
    if (i === 0) {
      return samples;
    } else {
      var x = Math.random();
      var y = Math.random();
      var z = Math.pow(x, 2) + Math.pow(y, 2) < 1 ? 1 : 0;
      return function() { return loop(i - 1, samples.concat([z])); };
    }
  };
  var add = function(a, b) { return a + b; };
  var s = loop(n, []);
  while (typeof s === 'function') { s = s(); }
  return s.reduce(add, 0) / n * 4;
};

var time = function(f) {
  var t0 = new Date();
  var value = f();
  return { value: value, elapsed: (new Date()) - t0 };
};

var postRequest = function(data, verbose) {
  var postData = querystring.stringify(data);
  var options = {
    hostname: 'docs.google.com',
    port: 443,
    path: '/forms/d/1kgcyHreL571NNkp3sx8HFTyXThx1XFHIPOqUoB4V2PQ/formResponse',
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Content-Length': postData.length
    }
  };

  var log = function(s) {
    if (verbose) {
      console.log(s);
    }
  };

  log(data);

  var req = https.request(options, function(res) {
    log('Status: ' + res.statusCode);
    //log('Headers: ' + JSON.stringify(res.headers));
    res.setEncoding('utf8');
    res.on('data', function (chunk) {
      //log('Body: ' + chunk);
    });
    res.on('end', function() {
      //log('No more data in response.');
    });
  });

  req.on('error', function(e) {
    log('Error: ' + e.message);
  });

  req.write(postData);
  req.end();
};

var recordResult = function(name, elapsed, norm, nodeVersion, commit, verbose) {
  postRequest({
    'entry.1021264083': name,
    'entry.1475555098': elapsed,
    'entry.983343629': norm,
    'entry.1570713488': nodeVersion,
    'entry.2058119347': commit,
    'entry.264740260': 'not-a-bot'
  }, verbose);
};

var onTravis = function() {
  return process.env.TRAVIS === 'true';
};

var travisNodeVersion = function() {
  //return process.env.TRAVIS_NODE_VERSION;
  return process.version;
};

var travisCommit = function() {
  return process.env.TRAVIS_COMMIT;
};

var main = function() {
  console.log('Running norm test.');
  var norm = time(function() {
    return estimatePI(1e5);
  }).elapsed;
  _.each(examples, function(example) {
    console.log('Running ' + example + '.');
    var result = time(function() {
      var value;
      webppl.run(loadExample(example), function(s, val) {
        assert(erp.isErp(val));
        value = val;
      });
      return value;
    });
    if (onTravis()) {
      console.log('Recording result.');
      recordResult(example, result.elapsed, norm, travisNodeVersion(), travisCommit(), true);
    } else {
      console.log(example, result.elapsed, norm);
    }
  });
};

main();
