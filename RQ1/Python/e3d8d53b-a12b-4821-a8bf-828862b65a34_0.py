override predicate isSink(DataFlow::Node sink) {
  exists(API::Node functionNode |
    sink = functionNode.getReturn().getAUse() and
    (
      functionNode = API::moduleImport("sklearn.model_selection").getMember("GroupKFold") or
      functionNode = API::moduleImport("sklearn.model_selection").getMember("GroupShuffleSplit") or
      functionNode = API::moduleImport("sklearn.model_selection").getMember("KFold") or
      functionNode = API::moduleImport("sklearn.model_selection").getMember("LeaveOneGroupOut") or
      //... include the remaining functions here in the same way ...
      functionNode = API::moduleImport("sklearn.model_selection").getMember("cross_val_score")
    )
  )
}
