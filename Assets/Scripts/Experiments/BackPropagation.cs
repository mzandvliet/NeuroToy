using Unity.Collections;
using UnityEngine;
using Ramjet.Mathematics;
using System.Collections.Generic;

/* 
    Todo:

    Flesh this out into a thing that lets you compose forward/backward nets and backprop them easily.

    Support for multi-in and multi-out

    Make backprop state separate, we don't always need or want it around.

    I think maybe I need to introduce slots and connections as concepts
*/

namespace BackPropPractice {
    public class BackPropagation : MonoBehaviour {

        private void Awake() {
            const int layerSize = 16;

            var a = new ConstNode(layerSize);
            Math.FillConstant(a.Values, 3f);
            var b = new ConstNode(layerSize);
            Math.FillConstant(b.Values, 2f);
            var op1 = new AddNode(a, b);

            var c = new ConstNode(layerSize);
            Math.FillConstant(c.Values, -5f);
            var op2 = new MultiplyNode(op1, c);

            var target = new ConstNode(layerSize);
            Math.FillConstant(target.Values, 15f);

            var loss = new LossNode(op2, target);

            

            Forward(loss);
            Debug.Log(op2.Values[0]);
            Backward(loss);


            const float rate = 0.01f;
            // a.Value += dLdA * rate;
            // b.Value += dLdB * rate;
            // c.Value += dLdC * rate;

            // Todo: easier disposal of graph
            a.Dispose();
            b.Dispose();
            op1.Dispose();
            c.Dispose();
            op2.Dispose();
            target.Dispose();
            loss.Dispose();
        }

        private static void Forward(IFloatNode network) {
            /* input node represents last output node in network
            we traverse it back-to-front, breadth-first, to
            /construct an ordered update list
            not the coolest or most correct, but will work */

            var nodeList = new List<IFloatNode>();

            var nodes = new Stack<IFloatNode>();
            nodes.Push(network);
            while (nodes.Count != 0) {
                var node = nodes.Pop();
                nodeList.Add(node);
                for (int i = 0; i < node.Inputs.Length; i++) {
                    nodes.Push(node.Inputs[i].Owner);
                }
            }

            nodeList.Reverse();

            for (int i = 0; i < nodeList.Count; i++) {
                nodeList[i].Forward();
            }
        }

        private static void Backward(IFloatNode network) {
            var nodes = new Stack<IFloatNode>();
            nodes.Push(network);
            while (nodes.Count != 0) {
                var node = nodes.Pop();
                node.Backward();
                for (int i = 0; i < node.Inputs.Length; i++) {
                    nodes.Push(node.Inputs[i].Owner);
                }
            }
        }
    }

    public interface IFloatNode : System.IDisposable {
        Slot[] Inputs { get; }
        Slot[] Outputs { get; }

        void Forward();
        void Backward();
    }

    public class Slot : System.IDisposable {
        public IFloatNode Owner { get; private set; }
        public Slot ConnectedSlot { get; private set; }


        public NativeArray<float> Values { get; set; }

        public NativeArray<float> Gradients { get; set; }

        public int Size {
            get { return Values.Length; }
        }

        public Slot(IFloatNode owner, int size) {
            Owner = owner;
            Values = new NativeArray<float>(size, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            Gradients = new NativeArray<float>(size, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        }

        public void Dispose() {
            Values.Dispose();
            Gradients.Dispose();
        }

        public bool IsConnected {
            get { return ConnectedSlot != null; }
        }

        public void Connect(Slot slot) {
            ConnectedSlot = slot;
        }
    }

    public abstract class AbstractNode : IFloatNode {
        public Slot[] Inputs { get; private set; }
        public Slot[] Outputs { get; private set; }

        public AbstractNode(int numInputs, int inputSize, int numOutputs, int outputSize) {
            Inputs = new Slot[numInputs];
            Outputs = new Slot[numOutputs];
            
            for (int i = 0; i < numInputs; i++) {
                Inputs[i] = new Slot(this, inputSize);
            }
            for (int i = 0; i < numOutputs; i++) {
                Outputs[i] = new Slot(this, outputSize);
            }
        }

        public void Dispose() {
            for (int i = 0; i < Inputs.Length; i++) {
                Inputs[i].Dispose();
            }
            for (int i = 0; i < Outputs.Length; i++) {
                Outputs[i].Dispose();
            }
        }

        public abstract void Forward();
        public abstract void Backward();
    }

    public class ConstNode : AbstractNode {
        public ConstNode(int size) : base (0, 0, 1, size) {
            
        }

        public override void Forward() {
            // Do nothing
        }

        public override void Backward() {
            // Do nothing
        }
    }

    public class AddNode : AbstractNode {
        public AddNode(IFloatNode a, IFloatNode b) : base(2, a.Outputs[0].Size, 1, a.Outputs[0].Size) {
            if (a.Outputs[0].Size != b.Outputs[0].Size) {
                throw new System.ArgumentException(string.Format("Input A size {0} is not equal to Input B size {1}", a.Outputs[0].Size, b.Outputs[0].Size));
            }

            Inputs[0].Connect(a.Outputs[0]);
            Inputs[1].Connect(b.Outputs[0]);
        }

        public override void Forward() {
            var values = Outputs[0].Values;
            for (int i = 0; i < Outputs[0].Values.Length; i++) {
                values[i] = Inputs[0].Values[i] + Inputs[1].Values[i];
            }
        }

        public override void Backward() {
            var grads = Inputs[0].Gradients
            for (int i = 0; i < Inputs[0].Size; i++) {
                Inputs[0].Gradients[i] = Outputs[0].Gradients[i];// * 1f;
            }

            for (int i = 0; i < Inputs[1].Size; i++) {
                Inputs[1].Gradients[i] = Outputs[0].Gradients[i];// * 1f;
            }
        }
    }

    public class MultiplyNode : IFloatNode {
        private NativeArray<float> _values;
        public NativeArray<float> Values { get { return _values; } }

        private NativeArray<float>[] _gradients;
        public NativeArray<float>[] Gradients { get { return _gradients; } }

        public IFloatNode[] Inputs { get; private set; }

        public MultiplyNode(IFloatNode a, IFloatNode b) {
            if (a.Values.Length != b.Values.Length) {
                throw new System.ArgumentException(string.Format("Input A size {0} is not equal to Input B size {1}", a.Values.Length, b.Values.Length));
            }
            _values = new NativeArray<float>(a.Values.Length, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            Inputs = new IFloatNode[2];
            Inputs[0] = a;
            Inputs[1] = b;
        }

        public void Dispose() {
            Values.Dispose();
            for (int i = 0; i < _gradients.Length; i++) {
                _gradients[i].Dispose();
            }
        }

        public void Forward() {
            for (int i = 0; i < Values.Length; i++) {
                _values[i] = Inputs[0].Values[i] * Inputs[1].Values[i];
            }
        }

        public void Backward(NativeArray<float> gradIn) {
            for (int i = 0; i < gradIn.Length; i++) {
                _gradients[0][i] = gradIn[i] * Inputs[1].Values[i];
            }

            for (int i = 0; i < gradIn.Length; i++) {
                _gradients[1][i] = gradIn[i] * Inputs[0].Values[i];
            }
        }
    }

    // Todo: also produce gradients w.r.t. target values.
    public class LossNode : IFloatNode {
        private NativeArray<float> _values;
        public NativeArray<float> Values { get { return _values; } }

        private NativeArray<float>[] _gradients;
        public NativeArray<float>[] Gradients { get { return _gradients; } }

        public IFloatNode[] Inputs { get; private set; }

        public LossNode(IFloatNode input, IFloatNode target) {
            if (input.Values.Length != target.Values.Length) {
                throw new System.ArgumentException(string.Format("Input size {0} is not equal to Target size {1}", input.Values.Length, target.Values.Length));
            }
            _values = new NativeArray<float>(input.Values.Length, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            Inputs = new IFloatNode[1];
            Inputs[0] = input;
            input.Output = this;
            Inputs[1] = target;

            _gradients = new NativeArray<float>[1];
            _gradients[0] = _values;
        }

        public void Dispose() {
            Values.Dispose();
            for (int i = 0; i < _gradients.Length; i++) {
                _gradients[i].Dispose();
            }
        }

        public void Forward() {
            for (int i = 0; i < Values.Length; i++) {
                _values[i] = Inputs[0].Values[i] + Inputs[1].Values[i];
            }
        }

        public void Backward(NativeArray<float> gradIn) {
            for (int i = 0; i < gradIn.Length; i++) {
                float gradient = Inputs[1].Values[i] - Inputs[0].Values[i];
                _gradients[0][i] = 0.5f * gradient * gradient;
            }
        }
    }
}
