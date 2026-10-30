"""Global admitted cohort with coverage of every participating source."""
from collections import Counter


def select(required_sources, rows, total=50):
    """Select stable evidence, requiring global uniqueness and source coverage.

    Rows must already have passed native admission and cross-revision novelty.
    This function does not certify either or replace any of those checks.
    """
    if (type(total) is not int or total < 1 or not required_sources
            or len(set(required_sources)) != len(required_sources) or len(required_sources) > total):
        raise ValueError('Distinct participating sources must fit the global quota')
    sources=set(required_sources);seen={};reports={};counts=Counter();validated=[]
    for row in rows:
        if row.get('source_token') not in sources:raise ValueError('Unexpected cohort source')
        if (row.get('trainable') is not True or row.get('status')!='training_sample_ready'
                or row.get('rendered') is not True or not row.get('sample_report')):
            raise ValueError('Only admitted rendered samples can enter a cohort')
        package=row.get('dataset_reference',{}).get('package_id')
        if not package:raise ValueError('Canonical package identity required')
        if package in seen:
            if seen[package]!=row['source_token']:raise ValueError('Package has ambiguous source identity')
            continue
        if row['sample_report'] in reports and reports[row['sample_report']]!=package:
            raise ValueError('Sample report represents multiple packages')
        reports[row['sample_report']]=package
        seen[package]=row['source_token'];counts[row['source_token']]+=1;validated.append(row)
    missing=[token for token in required_sources if not counts[token]]
    selected=[];used=set()
    if not missing:
        for token in required_sources:
            row=next(row for row in validated if row['source_token']==token)
            selected.append(row);used.add(row['dataset_reference']['package_id'])
        for row in validated:
            if len(selected)>=total:break
            package=row['dataset_reference']['package_id']
            if package not in used:selected.append(row);used.add(package)
    return dict(passed=not missing and len(selected)==total,required_total=total,
        minimum_per_source=1,participating_sources=list(required_sources),
        available_unique=len(validated),available_per_source={t:counts[t] for t in required_sources},
        missing_sources=missing,shortage=max(0,total-len(validated)),selected_rows=selected)
