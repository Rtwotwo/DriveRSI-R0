"""Training-only selection gate; diagnostic renderers remain available separately."""
from driversi.rsi.plugins.native_branch import prepare_select, run_select
from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence
from driversi.configuration import read_mapping


def execute(config, inputs, output, registry):
    result=run_select(config,inputs,output,registry)
    branch=read_mapping(result['artifacts']['report'])
    try:
        validate_trajectory_evidence(branch['trajectory_evidence'])
    except ValueError as error:
        result.update(status='rejected',reason=str(error))
        result['admission']={'trajectory':'rejected','visual':'not_started','trainable':False}
    return result


def register(registry, Provider):
    registry.register(Provider('qualified_native_branch_selection','1',
        (('world','navsim_teacher_execution'),),'selected_native_branch',prepare_select,execute,
        ('report','trajectory_window')))
