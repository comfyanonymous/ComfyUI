
def update_package_total_nodes(ddb_package_table, pacakge_id, total_nodes_value):
    if ddb_package_table is None or pacakge_id is None or total_nodes_value is None:
        print('🔴ddb_package_table is None')
        return None
    response = ddb_package_table.update_item(
        Key={
            'id': pacakge_id,
        },
        UpdateExpression='SET totalNodes = :val',
        ExpressionAttributeValues={
            ':val': total_nodes_value
        },
        ReturnValues="UPDATED_NEW"
    )

    print('ddb package item updated with totalNodes', '\n')
    return response