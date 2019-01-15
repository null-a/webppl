var tf = require('@tensorflow/tfjs-core');
require('@tensorflow/tfjs-node');

// f(x) = x^2 + 2x + 1

// f'(x) = 2x + 2

// var f = function(x) {
//   return tf.add(tf.square(x), tf.add(tf.mul(2.0, x), 1.0));
// };

// f(3.0).print(); // val = 16, grad = 8

// var g = tf.valueAndGrads(f);
// var x = g([tf.scalar(3.0)]);
// x.value.print();
// x.grads[0].print();

// tf.tidy(function() {
//   var x = tf.variable(tf.scalar(3.0));
//   // using this inside variableGrads doesn't work -- so i guess that
//   // evaluation f(x) doesn't build a graph. instead, the computation
//   // which one wants to differentiate has to happen within the
//   // function passed to variableGrads. in other words, i think the ad
//   // stuff is happening as a side effect of the main computation, but
//   // those side effects only happening within (the dynamic extent of)
//   // the function passed to variable grads.
//   var y = f(x);

//   var obj = tf.variableGrads(function() { return f(x); });
//   //console.log(obj);
//   obj.value.print()
//   obj.grads[0].print();

//   //console.log(tf.memory());
// });


//console.log(tf.memory());


// TODO: why doesn't this work as expected?

// This probably isn't what we want anyway. Ideally we'd like to avoid
// back prop through e.g. samplers entirely, not just back propagate
// zeros.

// var stopGradient = tf.customGrad(function(value) {
//   var gradFunc = function(dy) {
//     return tf.zerosLike(dy);
//   };
//   return {value, gradFunc};
// });
// var x = tf.valueAndGrad(function(x) { return stopGradient(tf.add(x, 0)); })(tf.tensor([1,2]));
// x.value.print();
// x.grad.print();
// throw 'halt';



// ========================================

// based on:
// https://github.com/tensorflow/tfjs-core/blob/ff514b9ecbeb9ec601216488147a7c484b5dc4cb/src/engine.ts#L474

var tape = require('@tensorflow/tfjs-core/dist/tape');
var engine = tf.ENV.engine;

// Object.defineProperty(tf.ENV.engine.__proto__, "getActiveTape", {
//   get: function() {
//     return this.activeTape;
//   },
//   enumerable: true,
//   configurable: true
// });


// this is an example of computing gradients without requiring
// computation to happen within a thunk. (the trampoline in webppl
// seems incompatible with the thunk approach.) this looks more the
// adnn than the public tf.js interface for computing gradients.

engine.startScope('gradients', true);

// const y = f();
var x = tf.variable(tf.scalar(3.0));
var y = tf.add(tf.square(x), tf.add(tf.mul(2.0, x), 1.0));

// Filter out the nodes that don't connect x => y.
// const filteredTape = getFilteredNodesXToY(this.activeTape, xs, y);
// var filteredTape = engine.activeTape;
var xs = [x];
var filteredTape = tape.getFilteredNodesXToY(engine.activeTape, xs, y);

// const accumulatedGradientMap: {[tensorId: number]: Tensor} = {};
// accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;
var accumulatedGradientMap = {};
accumulatedGradientMap[y.id] = tf.ones(y.shape);

// // Backprop gradients through the filtered nodes.
tape.backpropagateGradients(accumulatedGradientMap, filteredTape);

// const grads = xs.map(x => accumulatedGradientMap[x.id]);
// return {value: y, grads};
//console.log(accumulatedGradientMap);

// grad w.r.t x
console.log('====================');
console.log('value');
y.print();
console.log('grad');
accumulatedGradientMap[x.id].print();
console.log('====================');

//console.log(tf.memory());

// i looks like endScope avoids disposing of any tensors within the
// container `result``
var result = {}; // can be an object containing tensors. e.g. array, obj, combinations thereof
//var result = y;
engine.endScope(result, true);

//console.log(tf.memory());

//x.print(); // always works, never disposed of
//y.print(); // requires result (passed to `endScope`) to include y

// Q: which tensors are disposed of by `endScope`?
// Possible answer: Only tensors belonging to a tf.variable are retained.
