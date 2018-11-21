using System.Collections.Generic;
using System.Threading;
using NNBurst;
using Unity.Jobs;
using UnityEngine;
using Rng = Unity.Mathematics.Random;

/*

Todo:
Restore mutation and scoring
Restore genepool notion
 */

public class NeuralCreature {
    public Creature Body;
    public FCNetwork Mind;

    public Vector3 Goal;
    public float Score;
}

public class LocoApplication : MonoBehaviour {
    [SerializeField] private Terrain _terrain;

    [SerializeField] private int _populationSize = 64;
    [SerializeField] private int _episodeDuration = 250;
    [SerializeField] private float _mutationChance = 0.1f;
    [SerializeField] private float _mutationMagnitude = 0.1f;

    [SerializeField] private PhysicMaterial _physicsMaterial;

    private List<NeuralCreature> _creatures;
    private List<ITransform> _creatureInitPose;

    private bool _episodeActive;
    private int _episodeTickCount;
    private int _episodeCount;

    private Rng _rng = new Rng(1234);

    private FCNetwork _genotype;

    public static float EpisodeTime { get; private set; }

    private void Awake() {
        Application.runInBackground = true;

        var protoCreature = QuadrotorFactory.CreateCreature(
                Vector3.zero,
                Quaternion.identity);

        var config = new FCNetworkConfig();
        config.Layers.Add(new FCLayerConfig { NumNeurons = protoCreature.NumInputs });
        config.Layers.Add(new FCLayerConfig { NumNeurons = 30 });
        config.Layers.Add(new FCLayerConfig { NumNeurons = protoCreature.NumOutputs });
        
        Destroy(protoCreature.gameObject);

        // Create creature bodies
        _creatures = new List<NeuralCreature>(_populationSize);
        for (int i = 0; i < _populationSize; i++) {
            NeuralCreature nc = new NeuralCreature();

            Vector3 spawnPos = GetSpawnPosition(i, _populationSize);
            nc.Body = QuadrotorFactory.CreateCreature(
                spawnPos,
                GetSpawnRotation());

            nc.Mind = new FCNetwork(config);
            NeuralUtils.Initialize(nc.Mind, ref _rng);

            _creatures.Add(nc);
        }

        // Create a random genotype to seed the population
        _genotype = new FCNetwork(config);
        NeuralUtils.Initialize(_genotype, ref _rng);

        PrepareNewEpisode();
        MutatePopulation();

        // Store initial pose so we can reuse creature gameplay objects across tests
        _creatureInitPose = new List<ITransform>();
        CreatureFactory.SerializePose(_creatures[0].Body, _creatureInitPose);

        Physics.autoSimulation = false;
        Physics.autoSyncTransforms = false;

        var colliders = FindObjectsOfType<CapsuleCollider>();
        for (int i = 0; i < colliders.Length; i++) {
            // Bug this doesn't work, why?
            colliders[i].material = _physicsMaterial;
        }

        StartEpisode();
    }

    private void OnDestroy() {
        _genotype.Dispose();
        for (int i = 0; i < _creatures.Count; i++) {
            _creatures[i].Mind.Dispose();
        }
    }

    private void StartEpisode() {
        _episodeCount++;
        _episodeTickCount = 0;
        EpisodeTime = 0f;
        // Todo: Unify timer concepts
        _episodeActive = true;
    }

    private void OnGUI() {
        GUILayout.BeginVertical(GUI.skin.box, GUILayout.Width(200f));
        GUILayout.Label("Total ticks: " + _episodeCount * _episodeDuration);
        GUILayout.Label("Episode: " + _episodeCount);
        GUILayout.Label("Episode Tick: " + _episodeTickCount + "/" + _episodeDuration);
        GUILayout.Space(8);
        GUILayout.Label("Mutation Chance: " + _mutationChance);
        _mutationChance = GUILayout.HorizontalSlider(_mutationChance, 0.01f, 1f);
        GUILayout.Space(8);
        GUILayout.Label("Mutation Magnitude" + _mutationMagnitude);
        _mutationMagnitude = GUILayout.HorizontalSlider(_mutationMagnitude, 0.01f, 1f);
        GUILayout.Space(8);
        GUILayout.Label("Average Top Score" + _lastAverageTopScore);
        GUILayout.EndVertical();
    }

    private void Update() {
        EpisodeTime += Time.fixedDeltaTime;

        if (_episodeActive && _episodeTickCount < _episodeDuration) {
            // Simulate physics
            Physics.Simulate(0.02f);

            if (_episodeTickCount % 2 == 0) {
                // Update inputs
                for (int i = 0; i < _creatures.Count; i++) {
                    UpdateNetInputs(_creatures[i]);
                }

                // Singlethreaded update
                JobHandle h = new JobHandle();
                for (int i = 0; i < _creatures.Count; i++) {
                    h = JobHandle.CombineDependencies(h, NeuralJobs.ForwardPass(_creatures[i].Mind));
                }
                h.Complete();

                // Propagate network outputs to physics sim
                for (int i = 0; i < _creatures.Count; i++) {
                    UpdateNetOutputs(_creatures[i]);
                }
            }
            
            // Update score
            for (int i = 0; i < _creatures.Count; i++) {
                _creatures[i].Score += 
                    Vector3.Dot(_creatures[i].Goal, _creatures[i].Body.Root.Transform.forward) /
                    Mathf.Max(0.1f, _creatures[i].Body.Root.Rigidbody.angularVelocity.magnitude);
            }      

            _episodeTickCount++;
        }
        else {
            OnEpisodeComplete();
        }
        
    }

    private float _lastAverageTopScore;

    private void OnEpisodeComplete() {
        const int batchSize = 3;
        if (_episodeCount % batchSize == 0) {
            EvaluateAndUpdatePopulation();
        }

        PrepareNewEpisode();
        StartEpisode();
    }

    private void PrepareNewEpisode() {
        for (int i = 0; i < _creatures.Count; i++) {
            Vector3 spawnPos = GetSpawnPosition(i, _populationSize);
            CreatureFactory.RestorePose(
                _creatures[i].Body,
                spawnPos,
                GetSpawnRotation(),
                _creatureInitPose);
            Reset(_creatures[i]);
        }
    }

    private void EvaluateAndUpdatePopulation() {
        // Massage scores
        for (int i = 0; i < _creatures.Count; i++) {
            _creatures[i].Score = Mathf.Max(0f, _creatures[i].Score);
        }

        // Sort, take the top performing few
        _creatures.Sort((x, y) => x.Score.CompareTo(y.Score));
        float averageTopScore = 0f;
        int topRange = _populationSize / 16;
        List<NeuralCreature> topCreatures = _creatures.GetRange(_creatures.Count - topRange, topRange);
        for (int i = 0; i < topCreatures.Count; i++) {
            averageTopScore += topCreatures[i].Score;
        }
        averageTopScore /= (float) topRange;
        Debug.Log("Average top score: " + averageTopScore);

        // If score increased significantly, increase time per episode
        if (averageTopScore > _lastAverageTopScore) {
            _episodeDuration = Mathf.FloorToInt(_episodeDuration * 1.05f);
            _lastAverageTopScore = averageTopScore;

            Debug.Log("Time increase!");

            _mutationMagnitude = 0.1f;
            _mutationChance = 0.05f;
        }
        // If not, lower the bar slightly and increase mutation range to expand the search
        else {
            _lastAverageTopScore *= 0.995f;
            _mutationMagnitude = Mathf.Clamp01(_mutationMagnitude * 1.01f);
            _mutationChance = Mathf.Clamp01(_mutationChance * 1.01f);
        }

        ScoreBasedWeightedAverage(topCreatures, _genotype);

        MutatePopulation();
    }

    // Todo: reinstate
    private void MutatePopulation() {
        // Create new mutated brains from genotype
        // for (int i = 0; i < _creatures.Count; i++) {
        //     NetUtils.Copy(_genotype, _creatures[i].Mind);
        //     NetUtils.Mutate(
        //         _creatures[i].Mind,
        //         _mutationChance,
        //         _mutationMagnitude,
        //         ref _rng);
        // }
    }

    // Todo: reinstate
    public static void ScoreBasedWeightedAverage(List<NeuralCreature> nets, FCNetwork genotype) {
        // float scoreSum = 0f;

        // for (int i = 0; i < nets.Count; i++) {
        //     scoreSum += Mathf.Pow(nets[i].Score, 3f);
        // }

        // NNClassic.NetUtils.Zero(genotype);

        // for (int i = 0; i < nets.Count; i++) {
        //     float scale = Mathf.Pow(nets[i].Score, 3f) / scoreSum;
        //     Network candidate = nets[i].Mind;

        //     for (int l = 0; l < candidate.Layers.Count; l++) {
        //         for (int p = 0; p < candidate.Layers[l].ParamCount; p++) {
        //             genotype.Layers[l][p] += candidate.Layers[l][p] * scale;
        //         }
        //     }
        // }
    }

    private static void Reset(NeuralCreature c) {
        c.Score = 0f;
        c.Goal = UnityEngine.Random.insideUnitCircle.normalized;
        c.Goal = new Vector3(c.Goal.x, 0f, c.Goal.y);
        c.Body.Root.Transform.GetComponent<AngleSensor>().Goal = c.Goal;

        List<ISensor> sensors = c.Body.Sensors;
        for (int i = 0; i < sensors.Count; i++) {
            sensors[i].OnReset();
        }
    }

    private static void UpdateNetInputs(NeuralCreature c) {
        List<ISensor> sensors = c.Body.Sensors;
        int i = 0;
        for (int s = 0; s < sensors.Count; s++) {
            for (int sp = 0; sp < sensors[s].SensorCount; sp++) {
                c.Mind.Inputs[i] = sensors[s].Get(sp);
                i++;
            }
        }
    }

    private static void UpdateNetOutputs(NeuralCreature c) {
        List<IActuator> actuators = c.Body.Actuators;

        int i = 0;
        for (int s = 0; s < actuators.Count; s++) {
            for (int sp = 0; sp < actuators[s].ActuatorCount; sp++) {
                actuators[s].Set(sp, c.Mind.Last.Outputs[i]);
                i++;
            }
        }
    }

    // Todo: Derive spawn height from body type
    private static Vector3 GetSpawnPosition(int i, int max) {
        const float separation = 5f;
        return new Vector3(-max / 2 * separation + i * separation, 0.6f, 0f);
    }

    private static Quaternion GetSpawnRotation() {
        return Quaternion.identity;//Quaternion.Slerp(Quaternion.identity, UnityEngine.Random.rotation, 0.05f);
    }

    private void OnDrawGizmos() {
        if (!Application.isPlaying) {
            return;
        }

        Gizmos.color = Color.yellow;
        for (int i = 0; i < _creatures.Count; i++) {
            Gizmos.DrawRay(_creatures[i].Body.Root.Transform.position, _creatures[i].Goal);
        }
    }
}
