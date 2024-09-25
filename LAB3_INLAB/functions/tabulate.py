def tabulate(data, headers):
    table = []
    header_row = " | ".join(headers)
    table.append(header_row)
    table.append("-" * len(header_row))

    for row in data:
        table.append(" | ".join(map(str, row)))

    return "\n".join(table)
