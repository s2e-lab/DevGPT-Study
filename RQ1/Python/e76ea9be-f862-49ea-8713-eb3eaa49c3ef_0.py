class FeatureExtractor(nn.Module):
    """
    薬物と標的の両方のダミー特徴抽出器。
    通常、これは薬物の分子グラフのためのCNNとターゲットタンパク質配列のためのRNNの組み合わせである可能性があります。
    """
    ...

class DomainDiscriminator(nn.Module):
    """
    CDANモジュールのためのドメインディスクリミネータ。
    """
    ...

class DrugBAN(nn.Module):
    """
    基本的なDrugBANアーキテクチャ。
    """
    ...
