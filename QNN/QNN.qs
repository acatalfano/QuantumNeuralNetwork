namespace My.QNN {
    open Std.Math;

    import Std.Convert.IntAsDouble;
    import Std.StatePreparation.PreparePureStateD;
    import Std.Arrays.Reversed;

    @EntryPoint()
    operation QNN() : Unit {
        // TODO: stuff needs to happen here, in this main entry-point.
        //          ...Stuff bigger than "Hello quantum world!"


        Message("Hello quantum world!");
    }

    // TODO: when I try to use this type, it annoys the LittleEndian-oriented functions
    //          can I add the right wrappers to provide "LittleEndian vs BigEndian" type safety?
    //          both for OPERATORS and REGISTERS???
    // newtype LittleEndian = Qubit[];

    /// The full qnn circuit
    ///
    /// INPUT:
    ///        featureVector
    ///            - the preprocessed feature vector
    ///                (
    ///                    with input values repeated,
    ///                    padded with ancillary features,
    ///                    and length equal to an integral factor of 2
    ///                )
    ///        allLayersParams
    ///            - array of each convolutional layer's params
    ///        sampleCount
    ///            - the number of samples to take when approximating state probabilities
    /// ASSUMES:
    ///      length of every layer's theta is the same (in allLayersParams),
    ///      and complies with assumptions of LayerParams
    ///
    ///
    ///
    operation RunQNN (
        featureVector : Double[],
        allLayersParams : LayerParams[]
    ) : Int {

        // derive and reserve number of qubits required for amplitude encoding
        // (assumes featureVector is an integral length of a power of 2)
        let dblNumQubits = Lg(
            IntAsDouble(
                Length(featureVector)
            )
        );

        // issue with DoubleAsInt() when run through the python simulator
        let numQubits = Ceiling(dblNumQubits);

        use bigEndianQuantumRegister = Qubit[numQubits];
        // let qRegister = LittleEndian(qs);

        Message($"featureVector length {Length(featureVector)}");

        Message($"numQubits {numQubits}");
        // encode features as amplitudes
        PreparePureStateD(featureVector, bigEndianQuantumRegister);

        // apply the LayerParams
        for layerParams in allLayersParams {
            ConvolutionalLayer(layerParams, bigEndianQuantumRegister);
        }

        // measure (expects little-endian array of qubits)
        let result = MeasureInteger(Reversed(bigEndianQuantumRegister));

        // cleanup
        ResetAll(bigEndianQuantumRegister);

        // return measurement
        return result;
    }


    // type representing generic gate parameters
    newtype GateParams = (angle: Double, exp_beta: Double, exp_gamma: Double);

    /// Index pair representing index of a control qubit
    /// and the target qubit
    /// for a control gate
    ///
    newtype ControlTargetPair = (control: Int, target: Int);

    /// Type to encapsulate 1 QNN layer's parameters
    ///
    /// ASSUMES: controlGap <= Length(theta)
    ///          Length(controlledTheta) <= Length(theta)
    ///          Length(controlledTheta) is the expected length governed by
    ///
    newtype LayerParams = (
        theta: GateParams[],
        controlledTheta: GateParams[],
        controlGap: Int
    );


    function BuildControlTargetPairs (
        numQubits: Int,
        gapSize: Int
    ) : ControlTargetPair[] {

        mutable pairs = [];
        for idx in 1..(numQubits / GreatestCommonDivisorI(numQubits, gapSize)) {

            // pairs list + list containing single item, next ControlTargetPair
            set pairs +=
                [
                    ControlTargetPair(
                        (idx * gapSize) % numQubits,
                        (idx * gapSize - gapSize) % numQubits
                    )
                ];
        }

        return pairs;
    }



    /// A generic, parameterized gate
    /// with the expressive power of any quantum gate
    operation GenericGate (
        gateParams: GateParams,
        q: Qubit
    ) : Unit is Adj+Ctl {

        let angle = gateParams::angle;
        let exp_beta = gateParams::exp_beta;
        let exp_gamma = gateParams::exp_gamma;

        Message($"Generic gate with angle = {angle}, beta = {exp_beta}, gamma = {exp_gamma}");

        Rz(-(exp_beta + exp_gamma), q);
        Ry(-2.0 * angle, q);
        Rz(exp_gamma - exp_beta, q);
    }


    /// Build the convolution circuit
    /// theta:
    ///      list of GateParams
    ///      to be applied to GenericGates
    /// controlled_theta:
    ///      list of GateParams
    ///      to be applied to Controlled GenericGates
    /// control_gap:
    ///       the space b/w each control qubit
    ///       and the target qubit (w/ wrap around from qubit n-1 back to qubit 0)
    ///       ***larger gap means fewer control gates
    ///
    /// ASSUMES: length of theta is equal to number of qubits in qRegister
    ///
    operation ConvolutionalLayer (
        layerParams: LayerParams,
        //theta: GateParams[],
        //controlledTheta: GateParams[],
        //controlGap: Int,
        qs: Qubit[]
    ) : Unit is Adj+Ctl {

        let theta = layerParams::theta;
        let controlledTheta = layerParams::controlledTheta;
        let controlGap = layerParams::controlGap;

        for i in 0..Length(theta) - 1 {
            let (a,b,c) = theta[i]!;
            Message($"run w/ theta[{i}] = {a}, {b}, {c}");
        }

        for i in 0..Length(controlledTheta) - 1 {
            let (a,b,c) = controlledTheta[i]!;
            Message($"run w/ controlledTheta[{i}] = {a}, {b}, {c}");
        }
        // apply the single-qubit gates
        for idx in 0..Length(qs) - 1 {
            GenericGate(theta[idx], qs[idx]);
        }

        // apply the controlled 2-qubit gates
        let ctlTargPairs = BuildControlTargetPairs(Length(qs), controlGap);
        for idx in 0..Length(ctlTargPairs) - 1 {
            let ctlIndex  = ctlTargPairs[idx]::control;
            let targIndex = ctlTargPairs[idx]::target;

            Controlled GenericGate(
                [qs[ctlIndex]],
                (controlledTheta[idx], qs[targIndex])
            );
        }
    }
}
