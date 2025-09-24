from datarepo.core import Catalog, NlkDataFrame

__all__ = ["NlkDataFrame", "Catalog"]


def repl():
    """
        Starts an interactive python session with some of the datarepo imports.
    Allows for quick testing and inspection of data accessible via the datarepo
        client.
    """

    import IPython
    import polars as pl

    from datarepo.core import Catalog, Filter, NlkDataFrame

    print(
        r"""
------------------------------------------------

Welcome to
______      _                             
|  _  \    | |                            
| | | |__ _| |_ __ _ _ __ ___ _ __   ___  
| | | / _` | __/ _` | '__/ _ \ '_ \ / _ \ 
| |/ / (_| | || (_| | | |  __/ |_) | (_) |
|___/ \__,_|\__\__,_|_|  \___| .__/ \___/ 
                             | |          
                             |_|
------------------------------------------------

"""
    )

    IPython.start_ipython(
        colors="neutral",
        display_banner=False,
        user_ns={
            "Catalog": Catalog,
            "NlkDataFrame": NlkDataFrame,
            "Filter": Filter,
            "pl": pl,
        },
        argv=[],
    )
