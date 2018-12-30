using Unity.Collections;
using UnityEngine;
using Ramjet.Mathematics;
using System.Collections.Generic;

/* 
    Todo:

    Flesh this out into a thing that lets you compose forward/backward nets and backprop them easily.

    Make backprop state separate, we don't always need or want it around.
*/

namespace BackPropPractice {
    public class BackPropagation : MonoBehaviour {

        private void Awake() {
            const int layerSize = 16;

            var a = new ConstNode(layerSize);
            Math.FillConstant(a.Output, 3f);
            var b = new ConstNode(layerSize);
            Math.FillConstant(b.Output, 2f);
            var op1 = new AddNode(a, b);

            var c = new ConstNode(layerSize);
            Math.FillConstant(c.Output, -5f);
            var op2 = new MultiplyNode(op1, c);

            var t = new NativeArray<float>(layerSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Math.FillConstant(t, 15f);

            const float rate = 0.01f;

            // Todo: automatically traverse graph
            Forward(op2);

            Debug.Log(op2.Output[0]);

            // float dLdAdd2 = target - result; // Note: get from SumSquareLoss node
            
            // float dLdC = add2.BackwardB(dLdAdd2);
            // float dLdAdd1 = add2.BackwardA(dLdAdd2);

            // float dLdA = add1.BackwardA(dLdAdd1);
            // float dLdB = add1.BackwardB(dLdAdd1);

            // Debug.Log("(" + a.Value + " + " + b.Value + ") + " + c.Value + " = " + result);

            // a.Value += dLdA * rate;
            // b.Value += dLdB * rate;
            // c.Value += dLdC * rate;

            a.Dispose();
            b.Dispose();
            op1.Dispose();
            c.Dispose();
            op2.Dispose();
            t.Dispose();
        }

        private static void Forward(IFloatNode network) {
            // input node represents last output node in network
            // we traverse it back-to-front, breadth-first, to
            // construct an ordered update list
            // not the coolest or most correct, but will work

            var nodeList = new List<IFloatNode>();

            var nodes = new Stack<IFloatNode>();
            nodes.Push(network);
            while (nodes.Count != 0) {
                var n = nodes.Pop();
                nodeList.Add(n);
                for (int i = 0; i < n.Inputs.Length; i++) {
                    nodes.Push(n.Inputs[i]);
                }
            }

            nodeList.Reverse();

            for (int i = 0; i < nodeList.Count; i++) {
                nodeList[i].Forward();
            }
        }
    }

    public interface IFloatNode : System.IDisposable {
        NativeArray<float> Output { get; }

        IFloatNode[] Inputs { get; }

        void Forward();
    }

    public class ConstNode : IFloatNode {
        public NativeArray<float> Output { get; private set; }

        public IFloatNode[] Inputs { get; private set; }

        public ConstNode(int size) {
            Output = new NativeArray<float>(size, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            Inputs = new IFloatNode[0];
        }

        public void Dispose() {
            Output.Dispose();
        }

        public void Forward() {
            // Do nothing
        }
    }

    public class AddNode : IFloatNode {
        private NativeArray<float> _output;
        public NativeArray<float> Output { get { return _output; } }

        public IFloatNode[] Inputs { get; private set; }

        public AddNode(IFloatNode a, IFloatNode b) {
            if (a.Output.Length != b.Output.Length) {
                throw new System.ArgumentException(string.Format("Output size A {0} is not equal to output size B {1}", a.Output.Length, b.Output.Length));
            }
            _output = new NativeArray<float>(a.Output.Length, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            Inputs = new IFloatNode[2];
            Inputs[0] = a;
            Inputs[1] = b;
        }

        public void Dispose() {
            Output.Dispose();
        }

        public void Forward() {
            for (int i = 0; i < Output.Length; i++) {
                _output[i] = Inputs[0].Output[i] + Inputs[1].Output[i];
            }
        }

        // Todo: put these in separate nodes?

        public void BackwardA(NativeArray<float> gradIn, NativeArray<float> gradOut) {
            for (int i = 0; i < gradIn.Length; i++) {
                gradOut[i] = gradIn[i];// * 1f;
            }
        }

        public void BackwardB(NativeArray<float> gradIn, NativeArray<float> gradOut) {
            for (int i = 0; i < gradIn.Length; i++) {
                gradOut[i] = gradIn[i];// * 1f;
            }
        }
    }

    public class MultiplyNode : IFloatNode {
        private NativeArray<float> _output;
        public NativeArray<float> Output { get { return _output; } }

        public IFloatNode[] Inputs { get; private set; }

        public MultiplyNode(IFloatNode a, IFloatNode b) {
            if (a.Output.Length != b.Output.Length) {
                throw new System.ArgumentException(string.Format("Output size A {0} is not equal to output size B {1}", a.Output.Length, b.Output.Length));
            }
            _output = new NativeArray<float>(a.Output.Length, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            Inputs = new IFloatNode[2];
            Inputs[0] = a;
            Inputs[1] = b;
        }

        public void Dispose() {
            Output.Dispose();
        }

        public void Forward() {
            for (int i = 0; i < Output.Length; i++) {
                _output[i] = Inputs[0].Output[i] * Inputs[1].Output[i];
            }
        }

        public void BackwardA(NativeArray<float> gradIn, NativeArray<float> gradOut) {
            for (int i = 0; i < gradIn.Length; i++) {
                gradOut[i] = gradIn[i] * Inputs[1].Output[i];
            }
        }

        public void BackwardB(NativeArray<float> gradIn, NativeArray<float> gradOut) {
            for (int i = 0; i < gradIn.Length; i++) {
                gradOut[i] = gradIn[i] * Inputs[0].Output[i];
            }
        }
    }


    // public class SquareLossNode {
    //     public FloatNode A {
    //         get;
    //         private set;
    //     }

    //     public FloatNode B {
    //         get;
    //         private set;
    //     }

    //     public SquareLossNode(FloatNode a, FloatNode b) {
    //         A = a;
    //         B = b;
    //     }

    //     public float Forward() {
    //         return 0.5f * (A.Value - B.Value) * (A.Value - B.Value);
    //     }

    //     public float Backward() {
    //         return A.Value - B.Value;
    //     }
    // }
}
