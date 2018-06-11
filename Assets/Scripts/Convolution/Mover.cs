using UnityEngine;

public class Mover : MonoBehaviour {
    [SerializeField] private float _posMoiseScale = 10f;
    [SerializeField] private float _posNoiseFreq = 1f;
    [SerializeField] private float _rotMoiseScale = 10f;
    [SerializeField] private float _rotNoiseFreq = 1f;

    private Vector3 _startPos;

    private void Awake() {
        _startPos = transform.position;
    }

    private void Update() {
        transform.position = _startPos + new Vector3(
            Mathf.PerlinNoise(Time.time * _posNoiseFreq * 0.732982f, Time.time * _posNoiseFreq * 0.732982f) * _posMoiseScale,
            Mathf.PerlinNoise(Time.time * _posNoiseFreq * 0.503204f, Time.time * _posNoiseFreq * 0.503204f) * _posMoiseScale,
            Mathf.PerlinNoise(Time.time * _posNoiseFreq * 0.304982f, Time.time * _posNoiseFreq * 0.304982f) * _posMoiseScale);
        transform.rotation = Quaternion.identity * Quaternion.Euler(
            Mathf.PerlinNoise(Time.time * _posNoiseFreq * 0.732982f, Time.time * _rotNoiseFreq * 0.732982f) * Time.deltaTime * _rotMoiseScale,
            Mathf.PerlinNoise(Time.time * _posNoiseFreq * 0.503204f, Time.time * _rotNoiseFreq * 0.503204f) * Time.deltaTime * _rotMoiseScale,
            Mathf.PerlinNoise(Time.time * _posNoiseFreq * 0.304982f, Time.time * _rotNoiseFreq * 0.304982f) * Time.deltaTime * _rotMoiseScale);
    }
}