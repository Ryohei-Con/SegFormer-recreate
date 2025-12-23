### プロジェクトの概要
SegFormerの再現実装をした。<br>
以下の論文と公式実装を参考にし、SegFormerの特徴であるhierarchical transformer, Efficient-Self-Attention, MIX-FFNをコードに落とし込んだ。<br>
https://arxiv.org/pdf/2105.15203<br>
https://github.com/NVlabs/SegFormer/tree/master<br>

ただ、実際に動かしてパラメータを最適化するとこまではしていない。<br>

### 感想
Efficient-Self-Attentionは論文の数式をより発展させた形の効率化を公式ではしていたためそちらを優先させたが、論文の数式と公式実装がどうして数学的に同じなのかを理解するのに苦労した。<br>
パッチ化と二次元化を行き来して畳み込み層を挟むことでembeddingをしなくても良い点は興味深かった。<br>
この手順を踏むことで段階的に解像度を荒くして階層的なfeature mapをつくることができる点も面白かった。
