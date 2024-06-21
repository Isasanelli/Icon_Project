import rdflib
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF, XSD


def create_knowledge_base(data):
    g = Graph()

    for index, row in data.iterrows():
        patient_uri = URIRef( f"http://example.org/patient{index}" )
        g.add( (patient_uri, RDF.type, FOAF.Person) )
        g.add( (patient_uri, FOAF.name, Literal( f"Patient {index}", datatype=XSD.string )) )

        # Add other relevant data from the row
        if 'Age' in row:
            g.add( (patient_uri, URIRef( "http://example.org/age" ), Literal( row['Age'], datatype=XSD.integer )) )
        if 'Gender' in row:
            g.add(
                (patient_uri, URIRef( "http://example.org/gender" ), Literal( row['Gender'], datatype=XSD.integer )) )
        if 'Ethnicity' in row:
            g.add( (patient_uri, URIRef( "http://example.org/ethnicity" ),
                    Literal( row['Ethnicity'], datatype=XSD.integer )) )
        if 'SocioeconomicStatus' in row:
            g.add( (patient_uri, URIRef( "http://example.org/socioeconomicStatus" ),
                    Literal( row['SocioeconomicStatus'], datatype=XSD.integer )) )
        if 'FamilyHistoryDiabetes' in row:
            g.add( (patient_uri, URIRef( "http://example.org/familyHistoryDiabetes" ),
                    Literal( row['FamilyHistoryDiabetes'], datatype=XSD.integer )) )
        if 'BMI' in row:
            g.add( (patient_uri, URIRef( "http://example.org/bmi" ), Literal( row['BMI'], datatype=XSD.float )) )
        if 'FastingBloodSugar' in row:
            g.add( (patient_uri, URIRef( "http://example.org/fastingBloodSugar" ),
                    Literal( row['FastingBloodSugar'], datatype=XSD.float )) )
        if 'HbA1c' in row:
            g.add( (patient_uri, URIRef( "http://example.org/hba1c" ), Literal( row['HbA1c'], datatype=XSD.float )) )

    return g


def query_knowledge_base(g):
    query = """
    SELECT (COUNT(?s) AS ?numPersons) WHERE {
        ?s a foaf:Person .
    }
    """
    results = g.query( query )
    for row in results:
        print( f"Numero di persone: {row.numPersons}" )
