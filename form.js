//var http = require('http')
var https = require('https');
var querystring = require('querystring')

var postData = querystring.stringify({
  'entry.104543288' : Math.random().toString()
});

// https://docs.google.com/forms/d/1neAI2kvDoQCVkVqGGJxb2TSUcfIEmKxDQSRuaTNGYxs/formResponse


var options = {
  hostname: 'docs.google.com',
  port: 443,
  path: '/forms/d/1neAI2kvDoQCVkVqGGJxb2TSUcfIEmKxDQSRuaTNGYxs/formResponse',
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Content-Length': postData.length
  }
};

var req = https.request(options, function(res) {
  console.log('STATUS: ' + res.statusCode);
  console.log('HEADERS: ' + JSON.stringify(res.headers));
  res.setEncoding('utf8');
  res.on('data', function (chunk) {
    console.log('BODY: ' + chunk);
  });
  res.on('end', function() {
    console.log('No more data in response.')
  })
});

req.on('error', function(e) {
  console.log('problem with request: ' + e.message);
});

// write data to request body
req.write(postData);
req.end();
