using UnityEngine;
using System.Collections.Generic;
using Unity.Collections;

namespace NNBurst {
    public class NeuralNetRenderer : MonoBehaviour {
        [SerializeField] private Mesh _mesh;
        [SerializeField] private Material _mat;

        private List<Matrix4x4> _objects;


        private void Awake() {
            _objects = new List<Matrix4x4>();

            for (int i = 0; i < 512; i++) {
                var mat = Matrix4x4.TRS(
                    new Vector3(i,0f,0f),
                    Quaternion.identity,
                    Vector3.one
                );
                _objects.Add(mat);
            }
        }
        private void Update() {
            Graphics.DrawMeshInstanced(_mesh, 0, _mat, _objects);
            // Todo: probably need material property blocks for things like per-object color
        }
    }
}
