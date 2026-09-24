//! The encoder, graded against the reference that produced the fixture.
//!
//! `tests/fixtures/laya_tiny.py` builds a 2-layer ModernBERT inside
//! laya's OWN `DecisionModel`, seeds it, runs it once and records every
//! intermediate. This reproduces the encoder half of that run on
//! NdArray — no GPU, no model directory, seconds.
//!
//! The geometry is chosen so the things that go wrong quietly cannot:
//! `local_attention` is 4 over a 15-token sequence, so the sliding
//! band's edge falls inside it; layer 0 attends globally and layer 1
//! slides, so BOTH rotary bases turn in one pass; and the second row is
//! padded, so the key-padding mask has to be real.

use burn::tensor::{Tensor, TensorData};
use combs_classify::sequence::DecisionBatch;
use combs_formats::SafetensorsSource;
use combs_models::{EncoderGeometry, LayaModel, ModernBert, TypedDecisionModel};

type B = burn::backend::NdArray<f32>;

/// Layer-to-layer drift in an encoder is multiplicative, so the bound is
/// on the LAST thing compared and the earlier ones are printed beside
/// it: a port that is right at layer 0 and wrong at layer 1 says where
/// it went wrong, where a single pass/fail does not.
const TOL: f32 = 2e-3;

fn fixture() -> serde_json::Value {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya_tiny.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\nrun tests/fixtures/laya_tiny.py to make it",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("the fixture is json")
}

/// The fixture's weights and config, laid out as a model directory so
/// the config travels the same reader a real checkpoint does.
fn staged() -> std::path::PathBuf {
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let dir = std::env::temp_dir().join("combs-laya-tiny");
    std::fs::create_dir_all(&dir).expect("a staging directory");
    std::fs::copy(
        src.join("laya_tiny.safetensors"),
        dir.join("model.safetensors"),
    )
    .expect("weights");
    std::fs::copy(src.join("laya_tiny_config.json"), dir.join("config.json")).expect("config");
    std::fs::copy(
        src.join("laya_tiny_tokenizer.json"),
        dir.join("tokenizer.json"),
    )
    .expect("tokenizer");
    dir
}

/// A `[..., last]` nested array out of the fixture, flattened.
fn floats(v: &serde_json::Value) -> Vec<f32> {
    fn walk(v: &serde_json::Value, out: &mut Vec<f32>) {
        match v {
            serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            serde_json::Value::Number(n) => out.push(n.as_f64().expect("a number") as f32),
            other => panic!("not a number: {other}"),
        }
    }
    let mut out = Vec::new();
    walk(v, &mut out);
    out
}

/// Worst absolute gap, and where.
fn worst(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "shape");
    got.iter()
        .zip(want)
        .enumerate()
        .fold((0.0f32, 0), |(w, at), (i, (g, t))| {
            let d = (g - t).abs();
            if d > w {
                (d, i)
            } else {
                (w, at)
            }
        })
}

#[test]
fn the_encoder_reproduces_laya_layer_by_layer() {
    let fx = fixture();
    let dir = staged();
    let source = SafetensorsSource::load(&dir).expect("the staged checkpoint opens");
    let meta = combs_formats::ModelSource::metadata(&source);
    let cfg = meta
        .encoder
        .clone()
        .expect("the tiny config names an encoder");

    // The reader's own answers about this config, before any tensor runs.
    assert_eq!(
        cfg.local_window, 2,
        "HF states the whole 4; the kernel wants the radius"
    );
    assert!(cfg.is_global_layer(0), "layer 0 attends to everything");
    assert!(!cfg.is_global_layer(1), "layer 1 slides");
    assert!((cfg.global_rope_theta - 160000.0).abs() < 1e-6);
    assert!((cfg.local_rope_theta - 10000.0).abs() < 1e-6);
    assert!(!cfg.norm_bias && !cfg.attention_bias && !cfg.mlp_bias);

    let device = Default::default();
    let model = ModernBert::<B>::load(
        &source,
        &device,
        "encoder.",
        cfg,
        EncoderGeometry::from_metadata(meta),
    )
    .expect("the encoder loads");

    let ids_rows: Vec<Vec<i64>> = serde_json::from_value(fx["input_ids"].clone()).expect("ids");
    let att_rows: Vec<Vec<f32>> =
        serde_json::from_value(fx["attention_mask"].clone()).expect("mask");
    let (batch, seq) = (ids_rows.len(), ids_rows[0].len());
    let ids = Tensor::<B, 1, burn::tensor::Int>::from_data(
        TensorData::new(ids_rows.concat(), [batch * seq]),
        &device,
    )
    .reshape([batch, seq]);
    let attention =
        Tensor::<B, 1>::from_data(TensorData::new(att_rows.concat(), [batch * seq]), &device)
            .reshape([batch, seq]);

    // A tap is the number of layers APPLIED: 0 is the stream the first
    // layer reads (the fixture's `embedded`, which is HF's
    // `ModernBertEmbeddings` output, embedding norm included), 1 is
    // after layer 0, and 2 — this fixture's whole stack — is the stream
    // the final norm has not yet touched. `encoded` is that norm's
    // output, which is why tap 2 and `encoded` are different tensors.
    let (encoded, taps) = model
        .forward_taps(ids.clone(), &attention, &[0, 1, 2])
        .expect("three taps over a two-layer stack");
    assert_eq!(
        taps.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        [0, 1, 2],
        "taps come back in the caller's order"
    );

    for (name, got) in [
        ("embedded", taps[0].1.clone()),
        ("layer0", taps[1].1.clone()),
        ("layer1", taps[2].1.clone()),
        ("encoded", encoded),
    ] {
        let want = floats(&fx[name]);
        let have: Vec<f32> = got.into_data().to_vec().expect("f32");
        let (gap, at) = worst(&have, &want);
        println!(
            "  {name:<8} worst {gap:.3e} at {at} (got {:.6}, want {:.6})",
            have[at], want[at]
        );
        assert!(
            gap <= TOL,
            "{name}: worst gap {gap:.3e} at index {at} — got {:.6}, want {:.6} (bound {TOL:.0e})",
            have[at],
            want[at]
        );
    }
}

#[test]
fn the_sliding_band_is_a_band_and_not_a_triangle() {
    // The cheapest way this port could be wrong and still look right: a
    // sliding layer masked causally still produces plausible hidden
    // states. So the same sequence is run twice, once with a token
    // AFTER the one being read changed. On a band the earlier position
    // must move; under a causal mask it could not.
    let fx = fixture();
    let dir = staged();
    let source = SafetensorsSource::load(&dir).expect("open");
    let meta = combs_formats::ModelSource::metadata(&source);
    let cfg = meta.encoder.clone().expect("an encoder");
    let device = Default::default();
    let model = ModernBert::<B>::load(
        &source,
        &device,
        "encoder.",
        cfg,
        EncoderGeometry::from_metadata(meta),
    )
    .expect("loads");

    let mut rows: Vec<Vec<i64>> = serde_json::from_value(fx["input_ids"].clone()).expect("ids");
    let (batch, seq) = (rows.len(), rows[0].len());
    let att = Tensor::<B, 1>::from_data(
        TensorData::new(vec![1.0f32; batch * seq], [batch * seq]),
        &device,
    )
    .reshape([batch, seq]);
    let run = |rows: &Vec<Vec<i64>>| {
        let ids = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(rows.concat(), [batch * seq]),
            &device,
        )
        .reshape([batch, seq]);
        model
            .forward(ids, &att)
            .into_data()
            .to_vec::<f32>()
            .expect("f32")
    };
    let before = run(&rows);
    // Change the LAST token of row 0; read position 0's hidden state.
    rows[0][seq - 1] = 7;
    let after = run(&rows);
    let h = meta.hidden_size;
    let moved = (0..h)
        .map(|i| (before[i] - after[i]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 1e-4,
        "position 0 did not move when a LATER token changed: this encoder is attending causally, not in a band"
    );
}

#[test]
fn the_heads_reproduce_layas_logits_and_its_act_decision() {
    // The encoder test above holds the body. This holds everything laya
    // puts on top of it, and the order it puts it in: the type added at
    // every position, the head over the WHOLE sequence with the padding
    // mask, the markers gathered AFTER that head, and the act features
    // read from the RAW logits with the pooled vector taken post-head.
    // Each of those is a place a reasonable reading goes backwards
    // without changing any shape.
    let fx = fixture();
    let dir = staged();
    let source = SafetensorsSource::load(&dir).expect("the staged checkpoint opens");
    let device = Default::default();
    let head_layers = fx["geometry"]["head_layers"].as_u64().expect("head_layers") as usize;
    let meta = combs_formats::ModelSource::metadata(&source).clone();
    let model =
        LayaModel::<B>::load(&source, &device, meta, Some(head_layers)).expect("the model loads");

    let rows: Vec<Vec<i64>> = serde_json::from_value(fx["input_ids"].clone()).expect("ids");
    let att: Vec<Vec<u8>> = serde_json::from_value(fx["attention_mask"].clone()).expect("mask");
    let pos: Vec<Vec<u32>> = serde_json::from_value(fx["marker_pos"].clone()).expect("marker_pos");
    let msk: Vec<Vec<bool>> =
        serde_json::from_value(fx["marker_mask"].clone()).expect("marker_mask");
    let qt: Vec<u32> = serde_json::from_value(fx["qtype"].clone()).expect("qtype");

    let batch = DecisionBatch {
        rows: rows.len(),
        len: rows[0].len(),
        markers: pos[0].len(),
        input_ids: rows.iter().flatten().map(|&i| i as u32).collect(),
        attention: att.concat(),
        marker_pos: pos.concat(),
        marker_mask: msk.concat(),
        qtype: qt,
    };

    let out = model.decide(&batch).expect("it answers");

    let want_logits = floats(&fx["logits"]);
    let have_logits: Vec<f32> = out.logits.iter().flatten().copied().collect();
    let (gap, at) = worst(&have_logits, &want_logits);
    println!(
        "  logits   worst {gap:.3e} at {at} (got {:.5}, want {:.5})",
        have_logits[at], want_logits[at]
    );
    // The masked slot carries -1e4 in both, which is the check that a
    // marker nobody asked for scores nothing rather than scoring the
    // hidden state that happened to sit at position zero.
    assert!(
        want_logits.iter().any(|&l| l <= -1e3),
        "the fixture has no masked slot, so this proves nothing about one"
    );
    assert!(
        gap <= 5e-3,
        "logits: worst {gap:.3e} at {at} — got {}, want {}",
        have_logits[at],
        want_logits[at]
    );

    // The tempered readout, which is what a caller actually sees.
    let temperature: Vec<f32> =
        serde_json::from_value(fx["temperature"].clone()).expect("temperature");
    let want_probs: Vec<Vec<f32>> = serde_json::from_value(fx["probs"].clone()).expect("probs");
    for (r, want) in want_probs.iter().enumerate() {
        let k = want.len();
        let t = temperature[batch.qtype[r] as usize];
        let got = combs_classify::system_one::readout(&out.logits[r][..k], t);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4,
                "row {r} option {i}: p {g:.6} against laya's {w:.6}"
            );
        }
    }
    println!("  probs    reproduced at every option of every row");

    let want_act = floats(&fx["act"]);
    let have_act: Vec<f32> = out.act.iter().flatten().copied().collect();
    let (gap, at) = worst(&have_act, &want_act);
    println!(
        "  act      worst {gap:.3e} at {at} (got {:.5}, want {:.5})",
        have_act[at], want_act[at]
    );
    // A saturated act head agrees with any implementation; the fixture
    // is generated at a scale that keeps it off the rails, and this is
    // where that is checked rather than assumed.
    assert!(
        want_act.iter().all(|&a| (0.02..=0.98).contains(&a)),
        "the fixture's act head is saturated, so this bound proves little: {want_act:?}"
    );
    assert!(gap <= 1e-3, "act: worst {gap:.3e} at {at}");
}

/// `head_layers` came from the config alone and defaulted to 2 in
/// silence. The depth is a fact of the weights — `head.layers.{j}.*`
/// exists for exactly the layers there are — so the model reads it off
/// them, a config that agrees is fine, and one that disagrees is refused
/// naming both numbers. The tiny fixture has one head layer.
#[test]
fn the_heads_depth_comes_from_the_weights_and_a_wrong_config_is_refused() {
    let fx = fixture();
    let dir = staged();
    let source = SafetensorsSource::load(&dir).expect("the staged checkpoint opens");
    let device = Default::default();
    let meta = combs_formats::ModelSource::metadata(&source).clone();
    let stated = fx["geometry"]["head_layers"].as_u64().expect("head_layers") as usize;

    let model = LayaModel::<B>::load(&source, &device, meta.clone(), None)
        .expect("loads with no stated depth: the weights say");
    assert_eq!(
        model.head_layers(),
        stated,
        "the depth read off the weights is the fixture's"
    );

    LayaModel::<B>::load(&source, &device, meta.clone(), Some(stated))
        .expect("a config that agrees with the weights loads");

    let why = match LayaModel::<B>::load(&source, &device, meta, Some(stated + 1)) {
        Ok(_) => panic!(
            "a config stating {} head layers over {stated} loaded",
            stated + 1
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        why.contains(&format!("head_layers = {}", stated + 1))
            && why.contains(&format!("carry {stated}")),
        "the refusal names both numbers: {why}"
    );
}

/// The tap contract, on the tiny fixture: a tap index is the number of
/// layers applied, taps come back in the caller's order, a multi-tap
/// call is bitwise the same as the single-tap calls it stands for, the
/// deepest tap is NOT the normed output, and a tap past the stack is
/// refused by name rather than quietly absent — which is what the old
/// coordinate did with `taps=[2]` on this two-layer model.
#[test]
fn the_taps_coordinate_is_layers_applied_and_out_of_range_is_refused() {
    let dir = staged();
    let source = SafetensorsSource::load(&dir).expect("the staged checkpoint opens");
    let device = Default::default();
    let meta = combs_formats::ModelSource::metadata(&source).clone();
    let geom = EncoderGeometry::from_metadata(&meta);
    let cfg = meta.encoder.clone().expect("an encoder config");
    let encoder = ModernBert::<B>::load(&source, &device, combs_models::ENCODER_PREFIX, cfg, geom)
        .expect("the encoder loads");

    let fx = fixture();
    let ids_rows: Vec<Vec<i64>> = serde_json::from_value(fx["input_ids"].clone()).expect("ids");
    let att_rows: Vec<Vec<f32>> =
        serde_json::from_value(fx["attention_mask"].clone()).expect("mask");
    let (batch, seq) = (ids_rows.len(), ids_rows[0].len());
    let ids = Tensor::<B, 1, burn::tensor::Int>::from_data(
        TensorData::new(ids_rows.concat(), [batch * seq]),
        &device,
    )
    .reshape([batch, seq]);
    let attention =
        Tensor::<B, 1>::from_data(TensorData::new(att_rows.concat(), [batch * seq]), &device)
            .reshape([batch, seq]);
    let raw = |t: Tensor<B, 3>| -> Vec<f32> { t.into_data().to_vec().expect("f32") };

    // The caller's order, not ascending depth.
    let (_, out_of_order) = encoder
        .forward_taps(ids.clone(), &attention, &[2, 0])
        .expect("two taps in range");
    assert_eq!(
        out_of_order.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        [2, 0],
        "taps come back in the caller's order"
    );

    // A multi-tap call is bitwise the single-tap calls it stands for.
    for &t in &[0usize, 1, 2] {
        let (_, one) = encoder
            .forward_taps(ids.clone(), &attention, &[t])
            .expect("one tap in range");
        let want = raw(one[0].1.clone());
        let mine = out_of_order
            .iter()
            .find(|(i, _)| *i == t)
            .map(|(_, h)| raw(h.clone()));
        if let Some(mine) = mine {
            assert_eq!(
                mine, want,
                "tap {t} differs between a single and a multi call"
            );
        }
        let (final_out, all) = encoder
            .forward_taps(ids.clone(), &attention, &[0, 1, 2])
            .expect("three taps in range");
        assert_eq!(raw(all[t].1.clone()), want, "tap {t} in the full call");
        if t == 2 {
            assert_ne!(
                raw(all[2].1.clone()),
                raw(final_out),
                "the deepest tap is the stream BEFORE the final norm"
            );
        }
    }

    // One past the stack is refused by name, where it used to be absent.
    let why = encoder
        .forward_taps(ids, &attention, &[3])
        .expect_err("a tap past a two-layer stack is refused");
    let why = why.to_string();
    assert!(
        why.contains("tap 3") && why.contains("2-layer"),
        "the refusal names both: {why}"
    );
}
