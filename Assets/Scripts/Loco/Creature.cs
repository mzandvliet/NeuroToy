using System.Collections.Generic;
using UnityEngine;

public class CreaturePart {
    public Transform Transform;
    public Rigidbody Rigidbody;

    public CreaturePart Parent;
    public List<CreaturePart> Children;

    public CreaturePart() {
        Children = new List<CreaturePart>();
    }
}

// For now we still treat everything like bags of floats
// Each sensor/actuators provides a number of floats
// Todo: There is no need to make all of these actual components on gameobjects
public interface ISensor {
    int SensorCount { get; }
    float Get(int index);
    void OnReset();
}

public interface IActuator {
    int ActuatorCount { get; }
    void Set(int index, float value);
}

/* Todo:
 * Since all creatures of the same genus have equivalent addressing structure,
 * why should each phenotype have a copy of its adressing structure? It's
 * redundant. We shouldn't always have to, right?
 */
public class Creature : MonoBehaviour {
    public CreaturePart Root;
    public List<ISensor> Sensors;
    public List<IActuator> Actuators;

    public int NumInputs;
    public int NumOutputs;
}

public struct ITransform {
    public Vector3 Position;
    public Quaternion Rotation;

    public ITransform(Vector3 position, Quaternion rotation) {
        Position = position;
        Rotation = rotation;
    }
}

public static class CreatureFactory {
    private static readonly Stack<CreaturePart> Stack = new Stack<CreaturePart>(32);

    /*
     * Todo:
     * - pass in some params that decide hidden layer structure
     */
    public static void CreateNeuralTopology(Creature c) {
        c.Sensors = new List<ISensor>();
        c.Actuators = new List<IActuator>();

        Stack.Clear();
        Stack.Push(c.Root);

        int numInputs = 0;
        int numOutputs = 0;

        int sensorParamCount = 0;
        int actuatorParamCount = 0;
        while (Stack.Count > 0) {
            var part = Stack.Pop();

            var sensors = part.Transform.gameObject.GetComponentsInChildren<ISensor>();
            var actuators = part.Transform.gameObject.GetComponentsInChildren<IActuator>();

            // Print what we've got for a crappy way to match the visualization to inputs
            // Todo: Can't ask a sensor what it is, that's bad.
            Debug.Log("Sensors: ");
            for (int i = 0; i < sensors.Length; i++) {
                c.Sensors.Add(sensors[i]);
                if (sensors[i].SensorCount > 0) {
                    sensorParamCount += 1;
                    Behaviour b = (Behaviour) sensors[i];
                    Debug.Log(sensorParamCount + ". " + b.name + "." + b.GetType().Name + ": " + sensors[i].SensorCount);
                }
                numInputs += sensors[i].SensorCount;
            }
            Debug.Log("Actuators: ");
            for (int i = 0; i < actuators.Length; i++) {
                c.Actuators.Add(actuators[i]);
                if (actuators[i].ActuatorCount > 0) {
                    actuatorParamCount += 1;
                    Behaviour b = (Behaviour)actuators[i];
                    Debug.Log(actuatorParamCount + ". " + b.name + "." + b.GetType().Name + ": " + actuators[i].ActuatorCount);
                }
                numOutputs += actuators[i].ActuatorCount;
            }

            for (int i = 0; i < part.Children.Count; i++) {
                Stack.Push(part.Children[i]);
            }
        }

        c.NumInputs = numInputs;
        c.NumOutputs = numOutputs;
    }

    public static void SetActive(Creature creature, bool active) {
        Stack.Clear();

        Stack.Push(creature.Root);
        while (Stack.Count > 0) {
            var part = Stack.Pop();

            part.Transform.gameObject.SetActive(active);

            for (int i = 0; i < part.Children.Count; i++) {
                Stack.Push(part.Children[i]);
            }
        }
    }

    public static void SerializePose(Creature creature, List<ITransform> pose) {
        Stack.Clear();

        // Don't store root, we want to move it. Store children relative to root.
        for (int i = 0; i < creature.Root.Children.Count; i++) {
            Stack.Push(creature.Root.Children[i]);
        }

        Transform rootTrans = creature.Root.Transform;
        
        while (Stack.Count > 0) {
            var part = Stack.Pop();

            Vector3 localPos = rootTrans.InverseTransformPoint(part.Transform.position);
            Quaternion localRot = Quaternion.Inverse(rootTrans.rotation) * part.Transform.rotation;

            var trans = new ITransform(localPos, localRot);
            pose.Add(trans);
         
            for (int i = 0; i < part.Children.Count; i++) {
                Stack.Push(part.Children[i]);
            }
        }
    }

    public static void RestorePose(Creature creature, Vector3 rootPos, Quaternion rootRot, List<ITransform> pose) {
        SetActive(creature, false);

        Transform rootTrans = creature.Root.Transform;

        // Move root
        rootTrans.position = rootPos;
        rootTrans.rotation = rootRot;
        creature.Root.Rigidbody.velocity = Vector3.zero;
        creature.Root.Rigidbody.angularVelocity = Vector3.zero;

        Stack.Clear();
        for (int i = 0; i < creature.Root.Children.Count; i++) {
            Stack.Push(creature.Root.Children[i]);
        }

        // Move children relative to root
        int poseIdx = 0;
        while (Stack.Count > 0) {
            var part = Stack.Pop();

            part.Transform.position = rootTrans.TransformPoint(pose[poseIdx].Position);
            part.Transform.rotation = rootTrans.rotation * pose[poseIdx].Rotation;
            part.Rigidbody.velocity = Vector3.zero;
            part.Rigidbody.angularVelocity = Vector3.zero;

            poseIdx++;

            for (int i = 0; i < part.Children.Count; i++) {
                Stack.Push(part.Children[i]);
            }
        }

        SetActive(creature, true);
    }

    public static readonly string CreatureLayerName = "Creature";
}

public static class QuadrotorFactory {
    public static Creature CreateCreature(Vector3 position, Quaternion rotation) {
        var rootPart = CreateBody(position, rotation);
        AddRotors(rootPart);
        var creature = rootPart.Transform.gameObject.AddComponent<Creature>();
        creature.Root = rootPart;
        CreatureFactory.CreateNeuralTopology(creature);

        return creature;
    }

    private static CreaturePart CreateBody(Vector3 position, Quaternion rotation) {
        var torso = new CreaturePart();

        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = "Torso";
        go.transform.position = position;
        go.transform.rotation = rotation;
        go.transform.localScale = new Vector3(0.5f, 0.3f, 0.5f);
        go.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        torso.Transform = go.transform;

        torso.Rigidbody = go.AddComponent<Rigidbody>();
        torso.Rigidbody.mass = 5f;
        torso.Rigidbody.drag = 1f;
        torso.Rigidbody.angularDrag = 1f;
        torso.Rigidbody.useGravity = false;
        torso.Rigidbody.freezeRotation = true;
        torso.Rigidbody.constraints =
            RigidbodyConstraints.FreezePositionX |
            RigidbodyConstraints.FreezePositionY |
            RigidbodyConstraints.FreezePositionZ |
            RigidbodyConstraints.FreezeRotationX |
            RigidbodyConstraints.FreezeRotationZ;

//        go.AddComponent<CreatureCollider>();
        go.AddComponent<VelocitySensor>();
        go.AddComponent<AngleSensor>();

        return torso;
    }

    private static void AddRotors(CreaturePart parent) {
        AddRotor(parent, new Vector3(-0.5f, 0f, 0.5f), Quaternion.Euler(0f, -90f, 0f));
        AddRotor(parent, new Vector3( 0.5f, 0f, 0.5f), Quaternion.Euler(0f,  90f, 0f));
    }

    private static void AddRotor(CreaturePart parent, Vector3 localPosition, Quaternion localRotation) {
        var thrustObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        thrustObject.transform.position = parent.Transform.position + parent.Transform.rotation * localPosition;
        thrustObject.transform.rotation = parent.Transform.rotation * localRotation;
        thrustObject.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
        thrustObject.transform.parent = parent.Transform;
        var thruster = thrustObject.AddComponent<Thruster>();
        thruster.MaxForce = 10f;
        thruster.Rigidbody = parent.Rigidbody;
    }
}

public static class CrabFactory {
    public static Creature CreateCreature(Vector3 position, Quaternion rotation) {
        var rootPart = CreateTorso(position, rotation);
        AddLegs(rootPart);
        var creature = rootPart.Transform.gameObject.AddComponent<Creature>();
        creature.Root = rootPart;
        CreatureFactory.CreateNeuralTopology(creature);

        return creature;
    }

    private static CreaturePart CreateTorso(Vector3 position, Quaternion rotation) {
        var torso = new CreaturePart();

        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = "Torso";
        go.transform.position = position;
        go.transform.rotation = rotation;
        go.transform.localScale = new Vector3(0.6f, 0.6f, 0.6f);
        go.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        torso.Transform = go.transform;

        torso.Rigidbody = go.AddComponent<Rigidbody>();
        torso.Rigidbody.mass = 20f;
        torso.Rigidbody.drag = 0.05f;
        //torso.Rigidbody.inertiaTensor *= 2f;

//        torso.Rigidbody.isKinematic = true;

        go.AddComponent<CreatureCollider>();
        go.AddComponent<VelocitySensor>();
        go.AddComponent<AngleSensor>();

        return torso;
    }

    private static void AddLegs(CreaturePart parent) {
        const int numLegs = 4;
        const float angleBetweenLegs = 360f / numLegs;
        for (int i = 0; i < numLegs; i++) {
            Quaternion localRotation = Quaternion.Euler(0f, angleBetweenLegs * 0.5f + angleBetweenLegs * i, 0f);
            Vector3 localOffset = new Vector3(0f, 0f, 0.45f);
            Vector3 localPosition = localRotation * localOffset + Vector3.up * 0f;
            float legScaleMod = -0.05f * Vector3.Dot(Vector3.forward, localRotation * Vector3.forward);

            localRotation = localRotation * Quaternion.Euler(-20f, 0f, 0f);
            var legRoot = AddLegRootPart(parent, localPosition, localRotation, 0.45f + legScaleMod);
            var legPart = AddLegPart(legRoot, Quaternion.Euler(70f, 0f, 0f), 0.6f + legScaleMod);
        }
    }

    private static CreaturePart AddLegRootPart(CreaturePart parent, Vector3 localPosition, Quaternion localRotation, float length) {
        CreaturePart part = CreateCapsulePart("LegRoot", length, 0.15f);

        part.Parent = parent;
        parent.Children.Add(part);

        part.Transform.position = parent.Transform.position + localPosition;
        part.Transform.rotation = part.Parent.Transform.rotation * localRotation;
        part.Transform.gameObject.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        part.Rigidbody = part.Transform.gameObject.AddComponent<Rigidbody>();
        part.Rigidbody.mass = 10;

        var j = part.Transform.gameObject.AddComponent<ConfigurableJoint>();
        j.enablePreprocessing = false;
        j.connectedBody = parent.Transform.GetComponent<Rigidbody>();
        j.xMotion = ConfigurableJointMotion.Locked;
        j.yMotion = ConfigurableJointMotion.Locked;
        j.zMotion = ConfigurableJointMotion.Locked;
        j.angularXMotion = ConfigurableJointMotion.Limited;
        j.angularYMotion = ConfigurableJointMotion.Limited;
        j.angularZMotion = ConfigurableJointMotion.Limited;

        j.lowAngularXLimit = new SoftJointLimit() {
            limit = -40f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.highAngularXLimit = new SoftJointLimit() {
            limit = 30f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.angularYLimit = new SoftJointLimit() {
            limit = 50f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.angularZLimit = new SoftJointLimit() {
            limit = 30f,
            bounciness = 0f,
            contactDistance = 0f
        };

        j.rotationDriveMode = RotationDriveMode.Slerp;
        j.slerpDrive = new JointDrive() {
            positionSpring = 600,
            positionDamper = 20f,
            maximumForce = 9999999f
        };

        part.Transform.gameObject.AddComponent<CreatureJoint>();

        return part;
    }

    private static CreaturePart AddLegPart(CreaturePart parent, Quaternion rotationOffset, float length) {
        CreaturePart part = CreateCapsulePart("LegPart", length, 0.1f);

        part.Parent = parent;
        parent.Children.Add(part);

        part.Transform.position = GetCapsulePartTip(parent);
        part.Transform.rotation = part.Parent.Transform.rotation * rotationOffset;
        part.Transform.gameObject.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        part.Rigidbody = part.Transform.gameObject.AddComponent<Rigidbody>();
        part.Rigidbody.mass = 4;

        var j = part.Transform.gameObject.AddComponent<ConfigurableJoint>();
        j.enablePreprocessing = false;
        j.connectedBody = parent.Transform.GetComponent<Rigidbody>();
        j.xMotion = ConfigurableJointMotion.Locked;
        j.yMotion = ConfigurableJointMotion.Locked;
        j.zMotion = ConfigurableJointMotion.Locked;
        j.angularXMotion = ConfigurableJointMotion.Limited;
        j.angularYMotion = ConfigurableJointMotion.Locked;
        j.angularZMotion = ConfigurableJointMotion.Locked;

        j.lowAngularXLimit = new SoftJointLimit() {
            limit = -70f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.highAngularXLimit = new SoftJointLimit() {
            limit = 5f,
            bounciness = 0f,
            contactDistance = 0f
        };

        j.rotationDriveMode = RotationDriveMode.Slerp;
        j.slerpDrive = new JointDrive() {
            positionSpring = 500f,
            positionDamper = 20f,
            maximumForce = 9999999f
        };

        part.Transform.gameObject.AddComponent<CreatureJoint>();
        part.Transform.gameObject.AddComponent<CreatureCollider>();


//        var claw = part.Transform.gameObject.AddComponent<CreatureClaw>();
//        claw.Joint.anchor = new Vector3(0f, 0f, 0.64f);

        return part;
    }

    private static CreaturePart CreateCapsulePart(string name, float height, float radius) {
        CreaturePart part = new CreaturePart();
        part.Transform = new GameObject(name).transform;
        var c = part.Transform.gameObject.AddComponent<CapsuleCollider>();
        c.direction = 2;
        c.height = height;
        c.radius = radius;
        c.center = new Vector3(0f, 0f, c.height * 0.5f);

        // Add mesh renderer as child object
        var meshGo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        Object.Destroy(meshGo.GetComponent<CapsuleCollider>());
        meshGo.transform.parent = part.Transform;
        meshGo.transform.localScale = new Vector3(c.radius * 2f, c.height * 0.5f, c.radius * 2f);
        meshGo.transform.localPosition = Vector3.forward * c.height * 0.5f;
        meshGo.transform.localRotation = Quaternion.Euler(-90f, 0f, 0f);

        return part;
    }

    private static Vector3 GetCapsulePartTip(CreaturePart part) {
        var c = part.Transform.GetComponent<CapsuleCollider>();
        return part.Transform.TransformPoint(new Vector3(0f, 0f, c.height));
    }
}

public static class HumanoidFactory {
    public static Creature CreateCreature(Vector3 position, Quaternion rotation) {
        var rootPart = CreateTorso(position, rotation);
        AddLegs(rootPart);
        var creature = rootPart.Transform.gameObject.AddComponent<Creature>();
        creature.Root = rootPart;
        CreatureFactory.CreateNeuralTopology(creature);

        return creature;
    }

    private static CreaturePart CreateTorso(Vector3 position, Quaternion rotation) {
        var torso = new CreaturePart();

        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = "Torso";
        go.transform.position = position;
        go.transform.rotation = rotation;
        go.transform.localScale = new Vector3(0.3f, 0.3f, 0.3f);
        go.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        torso.Rigidbody = go.AddComponent<Rigidbody>();
        torso.Rigidbody.mass = 10f;
        torso.Rigidbody.drag = 0.1f;

        torso.Transform = go.transform;

//        go.AddComponent<CreatureCollider>();
//        go.AddComponent<VelocitySensor>();
//        go.AddComponent<AngleSensor>();

        return torso;
    }

    private static void AddLegs(CreaturePart parent) {
        {
            // Left leg
            Quaternion localRotation = Quaternion.Euler(90f, 0f, 0f);
            var legRoot = AddLegRootPart(parent, new Vector3(-0.15f, 0f, 0.0f), localRotation);
//            var legPart = AddLegPart(legRoot, Quaternion.Euler(0f, 0f, 0f));
            var foot = AddFoot(legRoot, Quaternion.Euler(-90f, 0f, 0f));
        }

        {
            // Right leg
            Quaternion localRotation = Quaternion.Euler(90f, 0f, 0f);
            var legRoot = AddLegRootPart(parent, new Vector3(0.15f, 0f, 0.0f), localRotation);
//            var legPart = AddLegPart(legRoot, Quaternion.Euler(0f, 0f, 0f));
            var foot = AddFoot(legRoot, Quaternion.Euler(-90f, 0f, 0f));
        }
    }

    private static CreaturePart AddLegRootPart(CreaturePart parent, Vector3 localPosition, Quaternion localRotation) {
        CreaturePart part = CreateCapsulePart("LegRoot", 0.45f, 0.1f);

        part.Parent = parent;
        parent.Children.Add(part);

        part.Transform.position = parent.Transform.position + parent.Transform.rotation * localPosition;
        part.Transform.rotation = part.Parent.Transform.rotation * localRotation;
        part.Transform.gameObject.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        part.Rigidbody = part.Transform.gameObject.AddComponent<Rigidbody>();
        part.Rigidbody.mass = 10;

        var j = part.Transform.gameObject.AddComponent<ConfigurableJoint>();
        j.enablePreprocessing = false;
        j.connectedBody = parent.Transform.GetComponent<Rigidbody>();
        j.xMotion = ConfigurableJointMotion.Locked;
        j.yMotion = ConfigurableJointMotion.Locked;
        j.zMotion = ConfigurableJointMotion.Locked;
        j.angularXMotion = ConfigurableJointMotion.Limited;
        j.angularYMotion = ConfigurableJointMotion.Locked;
        j.angularZMotion = ConfigurableJointMotion.Locked;

        j.highAngularXLimit = new SoftJointLimit() {
            limit = 80f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.lowAngularXLimit = new SoftJointLimit() {
            limit = -80f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.angularYLimit = new SoftJointLimit() {
            limit = 50f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.angularZLimit = new SoftJointLimit() {
            limit = 30f,
            bounciness = 0f,
            contactDistance = 0f
        };

        j.rotationDriveMode = RotationDriveMode.Slerp;
        j.slerpDrive = new JointDrive() {
            positionSpring = 600,
            positionDamper = 30,
            maximumForce = 9999999f
        };
      

        part.Transform.gameObject.AddComponent<CreatureJoint>();

        return part;
    }

    private static CreaturePart AddLegPart(CreaturePart parent, Quaternion rotationOffset) {
        CreaturePart part = CreateCapsulePart("LegPart", 0.2f, 0.05f);

        part.Parent = parent;
        parent.Children.Add(part);

        part.Transform.position = GetCapsulePartTip(parent);
        part.Transform.rotation = part.Parent.Transform.rotation * rotationOffset;
        part.Transform.gameObject.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        part.Rigidbody = part.Transform.gameObject.AddComponent<Rigidbody>();
        part.Rigidbody.mass = 7;

        var j = part.Transform.gameObject.AddComponent<ConfigurableJoint>();
        j.enablePreprocessing = false;
        j.connectedBody = parent.Transform.GetComponent<Rigidbody>();
        j.xMotion = ConfigurableJointMotion.Locked;
        j.yMotion = ConfigurableJointMotion.Locked;
        j.zMotion = ConfigurableJointMotion.Locked;
        j.angularXMotion = ConfigurableJointMotion.Limited;
        j.angularYMotion = ConfigurableJointMotion.Locked;
        j.angularZMotion = ConfigurableJointMotion.Locked;

        j.highAngularXLimit = new SoftJointLimit() {
            limit = 0f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.lowAngularXLimit = new SoftJointLimit() {
            limit = -80f,
            bounciness = 0f,
            contactDistance = 0f
        };

        j.rotationDriveMode = RotationDriveMode.Slerp;
        j.slerpDrive = new JointDrive() {
            positionSpring = 500,
            positionDamper = 25,
            maximumForce = 9999999f
        };

        part.Transform.gameObject.AddComponent<CreatureJoint>();

        return part;
    }

    private static CreaturePart AddFoot(CreaturePart parent, Quaternion rotationOffset) {
        var part = new CreaturePart();

        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = "Foot";
        go.transform.localScale = new Vector3(0.2f, 0.1f, 0.3f);
        go.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        part.Rigidbody = go.AddComponent<Rigidbody>();
        part.Rigidbody.mass = 4f;
        part.Rigidbody.drag = 0.1f;

        part.Transform = go.transform;

        part.Parent = parent;
        parent.Children.Add(part);

        part.Transform.position = GetCapsulePartTip(parent);
        part.Transform.rotation = part.Parent.Transform.rotation * rotationOffset;
        part.Transform.position += part.Transform.forward * 0.03f;
        part.Transform.gameObject.layer = LayerMask.NameToLayer(CreatureFactory.CreatureLayerName);

        var j = part.Transform.gameObject.AddComponent<ConfigurableJoint>();
        j.anchor = new Vector3(0f, 0.5f, -0.1f);
        j.enablePreprocessing = false;
        j.connectedBody = parent.Transform.GetComponent<Rigidbody>();
        j.xMotion = ConfigurableJointMotion.Locked;
        j.yMotion = ConfigurableJointMotion.Locked;
        j.zMotion = ConfigurableJointMotion.Locked;
        j.angularXMotion = ConfigurableJointMotion.Limited;
        j.angularYMotion = ConfigurableJointMotion.Locked;
        j.angularZMotion = ConfigurableJointMotion.Locked;

        j.highAngularXLimit = new SoftJointLimit() {
            limit = 40f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.lowAngularXLimit = new SoftJointLimit() {
            limit = -40f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.angularYLimit = new SoftJointLimit() {
            limit = 10f,
            bounciness = 0f,
            contactDistance = 0f
        };
        j.angularZLimit = new SoftJointLimit() {
            limit = 40f,
            bounciness = 0f,
            contactDistance = 0f
        };

        j.rotationDriveMode = RotationDriveMode.Slerp;
        j.slerpDrive = new JointDrive() {
            positionSpring = 400,
            positionDamper = 20,
            maximumForce = 9999999f
        };

//        part.Transform.gameObject.AddComponent<CreatureCollider>();
        part.Transform.gameObject.AddComponent<CreatureJoint>();

        return part;
    }

    private static CreaturePart CreateCapsulePart(string name, float height, float radius) {
        CreaturePart part = new CreaturePart();
        part.Transform = new GameObject(name).transform;
        var c = part.Transform.gameObject.AddComponent<CapsuleCollider>();
        c.direction = 2;
        c.height = height;
        c.radius = radius;
        c.center = new Vector3(0f, 0f, c.height * 0.5f);

        // Add mesh renderer as child object
        var meshGo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        Object.Destroy(meshGo.GetComponent<CapsuleCollider>());
        meshGo.transform.parent = part.Transform;
        meshGo.transform.localScale = new Vector3(c.radius * 2f, c.height * 0.5f, c.radius * 2f);
        meshGo.transform.localPosition = Vector3.forward * c.height * 0.5f;
        meshGo.transform.localRotation = Quaternion.Euler(-90f, 0f, 0f);

        return part;
    }

    private static Vector3 GetCapsulePartTip(CreaturePart part) {
        var c = part.Transform.GetComponent<CapsuleCollider>();
        return part.Transform.TransformPoint(new Vector3(0f, 0f, c.height));
    }
}