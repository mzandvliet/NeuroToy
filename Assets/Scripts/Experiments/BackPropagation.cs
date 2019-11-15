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
        private VariableManager _vars;

        private void Awake() {
            _vars = new VariableManager();
        }
        
    }

    public class Operation {
        
    }

    public class Layer {
        public Shape shape;
        public Variable variable;
    }

    public struct Shape {
        public int length;
    }

    public class Variable {
        public readonly string name;
        public readonly Shape shape;

        public Variable(string name, Shape shape) {
            this.name = name;
            this.shape = shape;
        }
    }

    public class VariableManager : System.IDisposable {
        private Dictionary<string, NativeArray<float>> _vars;

        public VariableManager() {
            _vars = new Dictionary<string, NativeArray<float>>();
        }

        public void Dispose() {
            foreach (var pair in _vars) {
                pair.Value.Dispose();
            }
        }

        public Variable Create(string name, Shape shape) {
            if (_vars.ContainsKey(name)) {
                throw new System.ArgumentException("Variable with name '{0}' already exists");
            }

            var variable = new Variable(name, shape);
            var matrix = new NativeArray<float>(shape.length, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            _vars.Add(variable.name, matrix);
            return variable;
        }
    }
}
