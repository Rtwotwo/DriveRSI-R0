"""Read trusted, sealed local caches in the isolated owned v2 namespace."""
import io
import pickle


class V2Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("navsim.v2."):
            module = "navsim." + module[len("navsim.v2."):]
        return super().find_class(module, name)


def load(handle):
    return V2Unpickler(handle).load()


def loads(data):
    return load(io.BytesIO(data))
