'use strict';
var assert = require('assert');
var Tensor = require('adnn/tensor');
var ad = require('adnn/ad');
var nn = require('adnn/nn');

function puts(x) {
  console.log(x);
  return x;
};

// Transpose.
// Do the conservative thing, and return a copy for now.
Tensor.prototype.T = function() {
  assert.ok(this.rank === 2);
  var h = this.dims[0];
  var w = this.dims[1];
  var y = new Tensor([w, h]);
  for (var i = 0; i < h; i++) {
    for (var j = 0; j < w; j++) {
      y.data[j * h + i] = this.data[i * w + j];
    }
  }
  return y;
};

ad.tensor.transpose = ad.newUnaryFunction({
  OutputType: Tensor,
  name: 'transpose',
  forward: function(a) {
    return a.T();
  },
  backward: function(a) {
    var h = this.x.dims[0];
    var w = this.x.dims[1];
    for (var i = 0; i < h; i++) {
      for (var j = 0; j < w; j++) {
        a.dx.data[j * h + i] += this.dx.data[i * w + j];
      }
    }
  }
});


// Matrix inverse.
// Ported from numeric.js.
Tensor.prototype.inv = function() {

  assert.ok(this.rank === 2);
  assert.ok(this.dims[0] === this.dims[1]);
  var n = this.dims[0];

  var Ai, Aj;
  var Ii, Ij;
  var i, j, k, x;

  var A = [];
  for (i = 0; i < n; i++) {
    Ai = new Float64Array(n);
    A.push(Ai);
    for (j = 0; j < n; j++) {
      Ai[j] = this.data[i * n + j];
    }
  }

  // Not using Float64 here as I want the convinience of passing I to
  // fromArray() which doesn't currently work with Float64Array.
  var I = [];
  for (i = 0; i < n; i++) {
    Ii = new Array(n);
    I.push(Ii);
    for (j = 0; j < n; j++) {
      Ii[j] = i === j ? 1 : 0;
    }
  }

  for(j = 0; j < n; ++j) {
    var i0 = -1;
    var v0 = -1;
    for(i = j; i !== n; ++i) {
      k = Math.abs(A[i][j]);
      if(k > v0) {
        i0 = i; v0 = k;
      }
    }
    Aj = A[i0];
    A[i0] = A[j];
    A[j] = Aj;
    Ij = I[i0];
    I[i0] = I[j];
    I[j] = Ij;
    x = Aj[j];
    for(k = j; k !== n; ++k) {
      Aj[k] /= x;
    }
    for(k = n - 1; k !== -1; --k) {
      Ij[k] /= x;
    }
    for(i = n - 1; i !== -1; --i) {
      if (i !== j) {
        Ai = A[i];
        Ii = I[i];
        x = Ai[j];
        for(k = j + 1; k !== n; ++k) {
          Ai[k] -= Aj[k] * x;
        }
        for(k = n - 1; k > 0; --k) {
          Ii[k] -= Ij[k] * x;
          --k;
          Ii[k] -= Ij[k] * x;
        }
        if(k===0) {
          Ii[0] -= Ij[0] * x;
        }
        }
    }
  }
  return new Tensor([n, n]).fromArray(I);
};

ad.tensor.inv = ad.newUnaryFunction({
  OutputType: Tensor,
  name: 'inverse',
  forward: function(A) {
    return A.inv();
  },
  backward: function(A) {
    var xT = this.x.T();
    A.dx = A.dx.add(xT.dot(this.dx).dot(xT).neg());
  }
});

// Determinant.
// Ported from numeric.js.
Tensor.prototype.det = function() {
  assert.ok(this.rank === 2);
  assert.ok(this.dims[0] === this.dims[1]);
  var n = this.dims[0];
  var ret = 1;

  var i, j, k;
  var Aj, Ai, alpha, temp, k1, k2, k3;

  var A = [];
  for (i = 0; i < n; i++) {
    Ai = new Float64Array(n);
    A.push(Ai);
    for (j = 0; j < n; j++) {
      Ai[j] = this.data[i * n + j];
    }
  }

  for(j = 0; j < n-1; j++) {
    k = j;
    for(i = j + 1; i < n; i++) {
      if(Math.abs(A[i][j]) > Math.abs(A[k][j])) {
        k = i;
      }
    }
    if(k !== j) {
      temp = A[k];
      A[k] = A[j];
      A[j] = temp;
      ret *= -1;
    }
    Aj = A[j];
    for(i = j + 1; i < n; i++) {
      Ai = A[i];
      alpha = Ai[j] / Aj[j];
      for(k = j + 1; k < n - 1; k += 2) {
        k1 = k + 1;
        Ai[k] -= Aj[k] * alpha;
        Ai[k1] -= Aj[k1] * alpha;
      }
      if (k !== n) {
        Ai[k] -= Aj[k] * alpha;
      }
    }
    if(Aj[j] === 0) {
      return 0;
    }
    ret *= Aj[j];
  }
  return ret * A[j][j];
};



ad.tensor.det = ad.newUnaryFunction({
  OutputType: Number,
  name: 'determinant',
  forward: function(A) {
    return A.det();
  },
  backward: function(A) {
    // A is square matrix.
    // Assume A is invertable.
    var n = A.x.dims[0];
    var invA = A.x.inv();
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        A.dx.data[i * n + j] += this.x * this.dx * invA.data[j * n + i];
      }
    }
  }
});


Tensor.prototype.dot = function(t) {

  var a = this, b = t;
  assert.ok(a.rank === 2 && b.rank === 2);
  assert.ok(a.dims[1] === b.dims[0]);

  var l = a.dims[1];
  var h = a.dims[0], w = b.dims[1];
  var y = new Tensor([h, w]);

  for (var r = 0; r < h; r++) {
    for (var c = 0; c < w; c++) {
      var z = 0;
      for (var i = 0; i < l; i++) {
        z += a.data[r * l + i] * b.data[w * i + c];
      }
      y.data[r * w + c] = z;
    }
  }
  return y;
};


ad.tensor.dot = ad.newBinaryFunction({
  OutputType: Tensor,
  name: 'dot',
  forward: function(a, b) {
    return a.dot(b);
  },
  backward1: function(A, B) {
    var Ap = ad.value(A);
    var Bp = ad.value(B);

    var Ah = Ap.dims[0];
    var Aw = Ap.dims[1];
    var Bh = Bp.dims[0];
    var Bw = Bp.dims[1];
    var hout = Ah;
    var wout = Bw;

    for (var l = 0; l < Ah; l++) {
      for (var m = 0; m < Aw; m++) {
        var z = 0;
        for (var j = 0; j < wout; j++) {
          z += this.dx.data[l * wout + j] * Bp.data[m * Bw + j];
        }
        A.dx.data[l * Aw + m] += z;
      }
    }
  },
  backward2: function(A, B) {
    var Ap = ad.value(A);
    var Bp = ad.value(B);

    var Ah = Ap.dims[0];
    var Aw = Ap.dims[1];
    var Bh = Bp.dims[0];
    var Bw = Bp.dims[1];
    var hout = Ah;
    var wout = Bw;

    for (var l = 0; l < Bh; l++) {
      for (var m = 0; m < Bw; m++) {
        var z = 0;
        for (var i = 0; i < Ah; i++) {
          z += this.dx.data[i * wout + m] * Ap.data[i * Aw + l];
        }
        B.dx.data[l * Bw + m] += z;
      }
    }

  }
});

var Vector = function(arr) {
  return new Tensor([arr.length, 1]).fromFlatArray(arr);
};

Tensor.prototype.inspect = Tensor.prototype.toString;


var test1 = function() {

  var X = ad.lift(new Tensor([2, 2]).fromArray([[2, 5], [.5, 1]]));

  // Y = log det X.T.dot(X)
  // Actual grad w.r.t. X:
  // 2 * pinv(X).T

  // Which at [[2, 5], [.5, 1]]
  // is:
  // [[-4, 2], [20, -8]]

  var Y = ad.scalar.log(ad.tensor.det(ad.tensor.dot(ad.tensor.transpose(X), X)));
  Y.backprop();
  puts(ad.value(Y));
  puts(ad.derivative(X));
};

//test1();
