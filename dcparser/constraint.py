import logging

operationsArr = ['<>', '<=', '>=', '=', '<', '>']
operationSign = ['IQ', 'LTE', 'GTE', 'EQ', 'LT', 'GT']


def is_symmetric(operation):
    if operation in set(['<>', '=']):
        return True
    return False


def contains_operation(string):
    """
    Method to check if a given string contains one of the operation signs.

    :param string: given string
    :return: operation index in list of pre-defined list of operations or
    Null if string does not contain any
    """
    for i in range(len(operationSign)):
        if string.find(operationSign[i]) != -1:
            return i
    return None


class DenialConstraint:
    """
    Class that defines the denial constraints.
    """
    def __init__(self, dc_string, schema):
        """
        Constructing denial constraint object.
        This class contains a list of predicates and the tuple_names which define a Denial Constraint

        :param dc_string: (str) string for denial constraint
        :param schema: (list[str]) list of attribute
        """
        dc_string = dc_string.replace('"', "'")
        split = dc_string.split('&')
        self.tuple_names = []
        self.predicates = []
        self.cnf_form = ""
        self.components = []

        # Find all tuple names used in DC
        logging.debug('DONE pre-processing constraint: %s', dc_string)
        for component in split:
            if contains_operation(component):
                break
            else:
                self.tuple_names.append(component)
        logging.debug('DONE extracting tuples from constraint: %s', dc_string)

        # Make a predicate for each component that's not a tuple name
        for i in range(len(self.tuple_names), len(split)):
            try:
                self.predicates.append(Predicate(split[i], self.tuple_names, schema))
            except Exception:
                logging.error('predicate %s', split[i])
                raise
        for p in self.predicates:
            self.components.append(p.components[0][1])

        # Create CNF form of the DC
        cnf_forms = [predicate.cnf_form for predicate in self.predicates]
        self.cnf_form = " AND ".join(cnf_forms)


class Predicate:
    """
    This class represents predicates.
    """
    def __init__(self, predicate_string, tuple_names, schema):
        """
        Constructing predicate object by setting self.cnf_form to e.g. t1."Attribute" = t2."Attribute".

        :param predicate_string: string shows the predicate
        :param tuple_names: name of tuples in denial constraint
        :param schema: list of attributes
        """
        self.schema = schema
        self.tuple_names = tuple_names
        self.cnf_form = ""
        op_index = contains_operation(predicate_string)
        if op_index is not None:
            self.operation_string = operationSign[op_index]
            self.operation = operationsArr[op_index]
        else:
            raise Exception('Cannot find operation in predicate')
        self.components = self.parse_components(predicate_string)
        for i in range(len(self.components)):
            component = self.components[i]
            if isinstance(component, str):
                self.cnf_form += component
            else:
                # Need to wrap column names in quotations for Postgres
                self.cnf_form += '{alias}."{attr}"'.format(
                        alias=component[0],
                        attr=component[1])
            if i < len(self.components) - 1:
                self.cnf_form += self.operation
        logging.debug("DONE parsing predicate: %s", predicate_string)

    def parse_components(self, predicate_string):
        """
        Parses the components of given predicate string
        Example: 'EQ(t1.ZipCode,t2.ZipCode)' returns [['t1', 'ZipCode'], ['t2','ZipCode']]

        :param predicate_string: predicate string
        :return: list of predicate components
        """
        # HC currently only supports DCs with two tuples per predicate
        # so raise an exception if a different number present
        num_tuples = len(predicate_string.split(','))
        if num_tuples < 2:
            raise Exception('Less than 2 tuples in predicate: ' +
                                    predicate_string)
        elif num_tuples > 2:
            raise Exception('More than 2 tuples in predicate: ' +
                                    predicate_string)

        operation = self.operation_string
        if predicate_string[0:len(operation)] != operation:
            raise Exception('First string in predicate is not operation ' + predicate_string)
        stack = []
        components = []
        current_component = []
        str_so_far = ""
        for i in range(len(operation), len(predicate_string)):
            str_so_far += predicate_string[i]
            if len(stack[-1:]) > 0 and stack[-1] == "'":
                if predicate_string[i] == "'":
                    if i == len(predicate_string) - 1 or \
                            predicate_string[i+1] != ')':
                        raise Exception("Expected ) after end of literal")
                    components.append(str_so_far)
                    current_component = []
                    stack.pop()
                    str_so_far = ""
            elif str_so_far == "'":
                stack.append("'")
            elif str_so_far == '(':
                str_so_far = ''
                stack.append('(')
            elif str_so_far == ')':
                if stack.pop() == '(':
                    str_so_far = ''
                    if len(stack) == 0:
                        break
                else:
                    raise Exception('Closed an unopened (' + predicate_string)
            elif predicate_string[i + 1] == '.':
                if str_so_far in self.tuple_names:
                    current_component.append(str_so_far)
                    str_so_far = ""
                else:
                    raise Exception('Tuple name ' + str_so_far + ' not defined in ' + predicate_string)

            elif (predicate_string[i + 1] == ',' or
                  predicate_string[i + 1] == ')') and \
                    predicate_string[i] != "'":

                # Attribute specified in DC not found in schema
                if str_so_far not in self.schema:
                    raise Exception('Attribute name {} not in schema: {}'.format(str_so_far, ",".join(self.schema)))

                current_component.append(str_so_far)
                str_so_far = ""
                components.append(current_component)
                current_component = []
            elif str_so_far == ',' or str_so_far == '.':
                str_so_far = ''
        return components

    def __str__(self):
        return self.cnf_form
