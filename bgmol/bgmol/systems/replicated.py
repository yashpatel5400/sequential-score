"""
Systems for batch evaluation.
"""

import copy
import numpy as np

from bgmol.util.importing import import_openmm
mm, unit, app = import_openmm()

from bgmol.util import BGMolException
from bgmol.systems import BaseSystem

__all__ = ["ReplicatedSystem"]


class ReplicatedSystem(BaseSystem):
    """
    Encapsules an openmm.System that contains multiple replicas of one system to enable batch computations.
    This class mimics the OpenMMSystem API. The implementation only works for specific forces, since
    forces of the replicated system have to be tailored so that the replicas are independent.

    Attributes
    ----------
    base_system : OpenMMSystem
        The base system that should be replicas.
    n_replicas : int
        Number of replicas to be stored in the replicated system.
    enable_energies : bool
        Whether to enable energy evaluations in batch. This option slows down the computation,
        since force objects have to be assigned to single replicas. This method enables energy
        evaluations via force groups (one force group per replica) but slows down force computations
        and propagation considerably. It also limits the maximal number of replicas to 32 (the max
        number of force groups OpenMM allows in one system). Therefore, `enable_energies=True` is not recommended.

    Notes
    -----
    Most methods in this class are static in order to enable conversion of single openmm objects (such as
    System, Topology, ...) as well as OpenMMSystem instances.

    Examples
    --------
    Replicate an openmm.System:
    >>> from openmmtools.testsystems import AlanineDipeptideImplicit
    >>> system = AlanineDipeptideImplicit().system
    >>> system_10batches = ReplicatedSystem.replicate_system(system, n_replicas=10, enable_energies=False)

    Replicate an bgmol.OpenMMSystem:
    >>> from bgmol import OpenMMToolsTestSystem
    >>> s = OpenMMToolsTestSystem("AlanineDipeptideImplicit")
    >>> s_10batches = ReplicatedSystem(s, n_replicas=10, enable_energies=False)
    >>> print(s_10batches.system, s_10batches.topology, s_10batches.positions)
    """

    def __init__(self, base_system: BaseSystem, n_replicas: int, enable_energies: bool=False):
        super(ReplicatedSystem, self).__init__()
        assert n_replicas > 0
        self._base_system = base_system
        # replicate
        self._system = self.replicate_system(base_system.system, n_replicas, enable_energies)
        self._topology = self.replicate_topology(base_system.topology, n_replicas)
        self._positions = self.replicate_positions(base_system.positions)
        # set system parameters
        for parameter, default in self._parameter_defaults.items():
            self.system_parameter(parameter, getattr(base_system, parameter), default)
        self.base_system_name = self.system_parameter("base_system_name", base_system.name, "")
        self.n_replicas = self.system_parameter("n_replicas", n_replicas, None)
        self.enable_energies = self.system_parameter("enable_energies", enable_energies, None)

    @property
    def system(self):
        return self._system

    @staticmethod
    def replicate_positions(positions):
        """Replicate particle positions."""
        # TODO
        if type(positions) is list:
            pass
        else:
            pass
        return NotImplemented

    @staticmethod
    def replicate_topology(base_topology: app.Topology, n_replicas: int):
        """Replicate an OpenMM Topology."""
        topology = app.Topology()
        # TODO
        return NotImplemented

    @staticmethod
    def replicate_system(base_system: mm.System, n_replicas: int, enable_energies: bool):
        """Replicate an OpenMM System."""
        system = mm.System()
        n_particles = base_system.getNumParticles()
        # particles
        for j in range(n_replicas):
            for i in range(n_particles):
                system.addParticle(base_system.getParticleMass(i))
                if system.isVirtualSite(i):
                    vs = system.getVirtualSite(i)
                    vs_copy = ReplicatedSystem._replicate_virtual_site(vs, n_particles, j)
                    system.setVirtualSite(i + j * n_particles, vs_copy)
        # constraints
        for j in range(n_replicas):
            for i in range(base_system.getNumConstraints()):
                p1, p2, distance = base_system.getConstraintParameters(i)
                system.addConstraint(p1 + j * n_particles, p2 + j * n_particles, distance)
        # properties
        system.setDefaultPeriodicBoxVectors(*(base_system.getDefaultPeriodicBoxVectors()))
        # forces
        for force in base_system.getForces():
            forcename = force.__class__.__name__
            methodname = f"_replicate_{forcename}"
            assert hasattr(ReplicatedSystem, methodname), f"Replicating {forcename} not implemented."
            replicate_force_method = getattr(ReplicatedSystem, methodname)
            replicated_forces = replicate_force_method(force, n_particles, n_replicas, enable_energies)
            for f in replicated_forces:
                system.addForce(f)
        return system

    @staticmethod
    def _replicate_virtual_site(vs, n_particles, replica):
        if isinstance(vs, mm.LocalCoordinatesSite):
            args = []
            for i in range(vs.getNumParticles()):
                args.append(vs.getParticle(i) + replica * n_particles)
            args.append(vs.getOriginWeights())
            args.append(vs.getXWeights())
            args.append(vs.getYWeights())
            args.append(vs.getLocalPosition())
            return mm.LocalCoordinatesSite(*args)
        elif isinstance(vs, mm.OutOfPlaneSite):
            args = []
            for i in range(vs.getNumParticles()):
                args.append(vs.getParticle(i) + replica * n_particles)
            args.append(vs.getWeight12())
            args.append(vs.getWeight13())
            args.append(vs.getWeightCross())
            return mm.OutOfPlaneSite(*args)
        elif isinstance(vs, mm.TwoParticleAverageSite):
            return mm.TwoParticleAverageSite(
                vs.getParticle(0) + replica * n_particles,
                vs.getParticle(1) + replica * n_particles,
                vs.getWeight(0),
                vs.getWeight(1)
            )
        elif isinstance(vs, mm.ThreeParticleAverageSite):
            return mm.ThreeParticleAverageSite(
                vs.getParticle(0) + replica * n_particles,
                vs.getParticle(1) + replica * n_particles,
                vs.getParticle(2) + replica * n_particles,
                vs.getWeight(0),
                vs.getWeight(1),
                vs.getWeight(2)
            )
        else:
            raise BGMolException(f"Unknown virtual site type: {type(vs)}.")

    @staticmethod
    def _replicate_HarmonicBondForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = mm.HarmonicBondForce()
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for j in range(n_replicas):
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                replicated_force.addBond(p1 + j * n_particles, p2 + j * n_particles, length, k)
            if enable_energies:
                # create a new force object for each replicate
                replicated_force.setForceGroup(j)
                replicated_forces.append(replicated_force)
                replicated_force = mm.HarmonicBondForce()
                replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_HarmonicAngleForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = mm.HarmonicAngleForce()
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for j in range(n_replicas):
            for i in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(i)
                replicated_force.addAngle(
                    p1 + j * n_particles,
                    p2 + j * n_particles,
                    p3 + j * n_particles,
                    angle, k)
            if enable_energies:
                # create a new force object for each replicate
                replicated_force.setForceGroup(j)
                replicated_forces.append(replicated_force)
                replicated_force = mm.HarmonicAngleForce()
                replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_PeriodicTorsionForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = mm.PeriodicTorsionForce()
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for j in range(n_replicas):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, angle, mult, k = force.getTorsionParameters(i)
                replicated_force.addTorsion(
                    p1 + j * n_particles,
                    p2 + j * n_particles,
                    p3 + j * n_particles,
                    p4 + j * n_particles,
                    angle, mult, k)
            if enable_energies:
                # create a new force object for each replicate
                replicated_force.setForceGroup(j)
                replicated_forces.append(replicated_force)
                replicated_force = mm.PeriodicTorsionForce()
                replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_CustomTorsionForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = mm.CustomTorsionForce(force.getEnergyFunction())
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for i in range(force.getNumGlobalParameters()):
            replicated_force.addGlobalParameter(force.getGlobalParameterName(i),
                    force.getGlobalParameterDefaultValue(i))
        for i in range(force.getNumPerTorsionParameters()):
            replicated_force.addPerTorsionParameter(force.getPerTorsionParameterName(i))
        for j in range(n_replicas):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, params = force.getTorsionParameters(i)
                replicated_force.addTorsion(p1 + j * n_particles,
                        p2 + j * n_particles,
                        p3 + j * n_particles,
                        p4 + j * n_particles,
                        params)
            if enable_energies:
                return NotImplemented
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_CMAPTorsionForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = mm.CMAPTorsionForce()
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for i in range(force.getNumMaps()):
            size, energy = force.getMapParameters(i)
            replicated_force.addMap(size, energy)
        for j in range(n_replicas):
            for i in range(force.getNumTorsions()):
                map_, *abs_ = force.getTorsionParameters(i)
                new_abs = [a_or_b + j * n_particles for a_or_b in abs_]
                replicated_force.addTorsion(map_, *new_abs)
            if enable_energies:
                return NotImplemented
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_NonbondedForce(force, n_particles, n_replicas, enable_energies):
        nonbonded_method = force.getNonbondedMethod()
        if nonbonded_method == mm.NonbondedForce.NoCutoff:
            return ReplicatedSystem._replicate_nonbonded_as_custom_bond_force(
                force,
                n_particles,
                n_replicas,
                enable_energies
            )
        else:
            return NotImplemented

    @staticmethod
    def _replicate_nonbonded_as_custom_bond_force(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        energy_string = "qiqj * ONE_4PI_EPS0 / r + 4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
        ONE_4PI_EPS0 = 138.935456

        def prep_force(force=force, energy_string=energy_string):
            f = mm.CustomBondForce(energy_string)
            f.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
            f.addPerBondParameter("qiqj")
            f.addPerBondParameter("epsilon")
            f.addPerBondParameter("sigma")
            return f

        replicated_force = prep_force()
        exceptions = {}
        for i in range(force.getNumExceptions()):
            p1, p2, qiqj, sigma, epsilon = force.getExceptionParameters(i)
            pair = (p1, p2) if p1 < p2 else (p2, p1)
            exceptions[pair] = (qiqj, sigma, epsilon)
        parameters = {}
        for i in range(force.getNumParticles()):
            q, sigma, epsilon = force.getParticleParameters(i)
            parameters[i] = (q, sigma, epsilon)
        assert force.getNumExceptionParameterOffsets() == 0
        assert force.getNumParticleParameterOffsets() == 0
        for j in range(n_replicas):
            for p1 in range(force.getNumParticles()):
                for p2 in range(p1+1, force.getNumParticles()):
                    if (p1,p2) in exceptions:
                        qiqj, sigma, epsilon = exceptions[(p1, p2)]
                        if (
                                (abs(qiqj.value_in_unit_system(unit.md_unit_system)) > 1e-10)
                                or
                                (abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 1e-10)
                        ):
                            replicated_force.addBond(p1 + j*n_particles, p2 + j*n_particles, [qiqj, epsilon, sigma])
                    else:
                        q1, sigma1, epsilon1 = parameters[p1]
                        q2, sigma2, epsilon2 = parameters[p2]
                        qiqj = q1*q2
                        sigma = 0.5 * (sigma1 + sigma2)
                        epsilon = np.sqrt(epsilon1 * epsilon2)
                        if (
                                (abs(qiqj.value_in_unit_system(unit.md_unit_system)) > 1e-10)
                                or
                                (abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 1e-10)
                        ):
                            replicated_force.addBond(p1 + j*n_particles, p2 + j*n_particles, [qiqj, epsilon, sigma])
            if enable_energies:
                replicated_force.setForceGroup(j)
                replicated_forces.append(replicated_force)
                replicated_force = prep_force()

        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_CMMotionRemover(force, n_particles, n_replicas, enable_energies):
        return []

    @staticmethod
    def _replicate_CustomGBForce(force, n_particles, n_replicas, enable_energies):
        raise NotImplementedError()

