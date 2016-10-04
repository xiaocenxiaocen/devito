from mpi4py import MPI

class Worker:
    def __init__(self, comm, ClassOperator):
        self.comm = comm
        self.ClassOperator = ClassOperator

    def run(self):
        ''' The worker spins on work farmed out to it until it is terminated.
        '''
        while True:
            model, shot, sourceind = self.comm.recv()
            operator = self.ClassOperator(model, shot, source)

            # Depending on what ClassOperator is, this may return
            # either (f, ) or (f, g) where f is the residule (scalar),
            # g is the gradient and dd is the linearised data.
            result = operator.run()
            print('Got result for worker ')
            self.comm.send(result, dest=0)


class Master:
    def __init__(self, comm, model, shots, source):
        self.comm = comm
        self.model = model
        self.shots = shots
        self.source = source

        self.nranks = self.comm.Get_size()
        self.nshots = shots.get_number_of_shots()
        assert(self.nranks - 1 <= self.nshots)

    def push_work(self, rank, shot, source):
        self.comm.send((self.model, shot, source), dest=rank)

    def run(self, worklist):
        status = MPI.Status()

        # Farm out tasks/shots.
        result = None
        source = None
        if self.source:
            source = self.source.get_shot(0)
        for rank, shot_id in zip(range(1, self.nranks), worklist):
            self.push_work(rank,
                           self.shots.get_shot(shot_id),
                           source)

        # Perform reduction operator
        for rank in range(1, self.nranks):
            _result = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
            if result is None:
                result = _result
            else:
                result = map(add, result, _result)
            print('Got result for worker ', rank, ' over', self.nranks-1)
        return result